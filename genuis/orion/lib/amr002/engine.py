# -*- coding: utf-8 -*-
"""
AlphaEngine — RL 训练引擎 (torch 张量模式)。

与 lib/engine.py 不同, 本模块使用 StackVM 在 torch 张量上直接执行公式,
不依赖 ultron 的 calc_factor。数据以 [N_assets, n_features, T_steps]
张量形式注入, 表达式以 token 序列形式生成。
"""
import pdb
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from .config import ModelConfig
from .vm import StackVM
from lib.cms003.metrics import Metrics

import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class AlphaEngine:
    """基于 torch 张量的特征挖掘训练引擎。

    Parameters
    ----------
    model : AlphaGPT
        Transformer 模型。
    feat_tensor : torch.Tensor
        [N_assets, n_features, T_steps] 的特征张量。
    target_ret : torch.Tensor
        [N_assets, T_steps] 的目标收益张量。
    features_list : list[str]
        特征名列表, 用于可读公式输出。
    """

    def __init__(self, model, feat_tensor, target_ret, features_list):
        self.model = model
        self.feat_tensor = feat_tensor
        self.target_ret = target_ret
        self.features_list = list(features_list)
        
        # 提取时间与资产特征以备构建 DataFrame (需要外部调用时赋予 model, 但 engine 里没有直接接触 assets 和 times, 我们可以自己造假的 或者 修改参数传入)
        # 这里最安全的做法是在 engine 初始化时接收 times 和 assets 数组，但为了保持最小改动，我们直接转为 NumPy 给 Metrics 用，Index 可以用自动生存的
        self.T, self.N = target_ret.shape[1], target_ret.shape[0]
        self.ret_df = pd.DataFrame(target_ret.cpu().numpy().T) # 形状: [T, N]

        self.vm = StackVM(n_features=len(features_list))
        self.opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        self.best_score = -float('inf')
        self.best_formula = None
        self.best_expression = None
        
        # [修改] 用于保存所有发现的有效公式
        self.discovery_log = []
        self.seen_formulas = set()

    # ---------------------------------------------------------------
    def compute_reward(self, formula_tokens):
        """计算单个公式的 reward。

        流程:
        1. StackVM 执行公式 → 因子张量 [N, T]
        2. 信号处理 + 简易回测
        3. 返回 fitness 分数

        Returns
        -------
        float
        """
        min_ic_threshold = 0.01
        res = self.vm.execute(formula_tokens, self.feat_tensor)
        if res is None:
            return 0.0
        
        try:
            factor_np = res.cpu().numpy()
            
            # 转成与 ret_df 同形状的 DataFrame: 行是时间(T), 列是资产(N)
            factor_df = pd.DataFrame(factor_np.T)
            
            # 调用 cms003.Metrics.quick
            eval_res = Metrics.quick(
                returns=self.ret_df,
                factors=factor_df,
                hold=1,
                skip=0,
                category=1, # EXCESS
                show_log=False
            )
            
            ic = eval_res.get('ic', np.nan)
            icir = eval_res.get('icir', np.nan)
            turnover = eval_res.get('turnover', np.nan)
            
            if np.isnan(ic) or np.isnan(icir):
                return 0.0
            
            ## 使用IC 为奖励
            if turnover > 1.0:
                return 0.0
            
            return abs(ic)

        except Exception as e:
            return -5.0

    # ---------------------------------------------------------------
    def train(self, train_steps=None, batch_size=None):
        """RL REINFORCE 训练循环。"""
        steps = train_steps or ModelConfig.TRAIN_STEPS
        bs = batch_size or ModelConfig.BATCH_SIZE

        logger.info(f"🚀 Starting Alpha Mining (tensor mode) | "
                    f"steps={steps}, batch={bs}")
        pbar = tqdm(range(steps), desc='Alpha Mining')

        for step in pbar:
            # ---- 1. 生成 token 序列 ----
            inp = torch.zeros(
                (bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            log_probs = []
            tokens_list = []

            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

            seqs = torch.stack(tokens_list, dim=1)

            # ---- 2. 执行 + 计算 reward ----
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)

            for i in range(bs):
                token_ids = seqs[i].tolist()
                score = self.compute_reward(token_ids)
                rewards[i] = score

                # [修改] 不再仅限于大于 0。
                # 只要是个在数学上合法的表达式 (expression is not None)，就把它保存下来！
                # 这样即使初期模型没找到高分因子，也能看清它到底在生成什么公式
                expression = self.vm.decode_readable(token_ids, self.features_list)
                if expression is not None:
                    formula_tuple = tuple(token_ids)
                    if formula_tuple not in self.seen_formulas:
                        self.seen_formulas.add(formula_tuple)
                        self.discovery_log.append({
                            'step': step,
                            'score': score,
                            'expression': expression,
                            'tokens': token_ids,
                        })
                
                # 依然追踪全局最高分用于打印
                if score > self.best_score:
                    self.best_score = score
                    self.best_formula = token_ids
                    self.best_expression = self.vm.decode_readable(token_ids, self.features_list)
                    tqdm.write(f"[!] New Best: Score {score:.4f} | Expr: {self.best_expression}")

            # ---- 3. REINFORCE 更新 ----
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            loss = torch.zeros(1, device=ModelConfig.DEVICE)
            for t in range(len(log_probs)):
                loss += (-log_probs[t] * adv).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # ---- 4. 进度条 ----
            valid_count = (rewards > -4.0).sum().item()
            pbar.set_postfix({
                'AvgRew': f'{rewards.mean().item():.2f}',
                'Best': f'{self.best_score:.4f}',
                'Valid': f'{valid_count}/{bs}',
            })

        logger.info(
            f"训练完成 | Best Score: {self.best_score:.4f} | "
            f"Best Expr: {self.best_expression}")
            
        # [修改] 训练结束时，按分数的“绝对值”从高到低排序 (因为高度负相关的因子同样有价值)
        self.discovery_log.sort(key=lambda x: abs(x['score']), reverse=True)
        top_discoveries = self.discovery_log[:50]
        
        return top_discoveries
