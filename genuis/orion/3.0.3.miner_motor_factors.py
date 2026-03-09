# -*- coding: utf-8 -*-
"""
3.0.3 特征挖掘引擎 — 基于 torch 张量 StackVM 模式

使用 lib1 的 AlphaGPT + StackVM, 在 GPU 张量上直接执行公式。
数据加载方式与 3.0.2 一致 (feather 文件), 转换为张量后注入引擎。
"""
import pdb, hashlib
import os, datetime, json, torch
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from lumina.genetic.util import create_id
from kdutils.macro2 import base_path
from kdutils.tactix import Tactix
from kdutils.logger import logger
from lib.amr002.config import ModelConfig
from lib.amr002.alphagpt import AlphaGPT
from lib.amr002.engine import AlphaEngine


# ── 数据加载 ──────────────────────────────────────────────
def load_data(method, task_id, ret_name):
    """加载 feather 数据并返回 DataFrame + 特征列名。"""
    
    base_dir1 = os.path.join(base_path, method, 'base', str(task_id))
    logger.info(f"load data {base_dir1}")
    
    train_factors = pd.read_feather(
        os.path.join(base_dir1, "train_data.feather"))
    val_factors = pd.read_feather(os.path.join(base_dir1, "val_data.feather"))

    train_return = pd.read_feather(
        os.path.join(base_dir1, "train_return.feather"))
    val_return = pd.read_feather(os.path.join(base_dir1, "val_return.feather"))

    train_factors['trade_time'] = pd.to_datetime(train_factors['trade_time'])
    val_factors['trade_time'] = pd.to_datetime(val_factors['trade_time'])
    train_return['trade_time'] = pd.to_datetime(train_return['trade_time'])
    val_return['trade_time'] = pd.to_datetime(val_return['trade_time'])

    total_factors = pd.concat([train_factors, val_factors], axis=0)
    total_return = pd.concat([train_return, val_return], axis=0)

    factors_cols = [
        col for col in total_factors.columns if col not in [
            'trade_time', 'code', 'time_weight', 'equal_weight',
            'f_funding_rate', 'f_funding_interval', 'nxt1_ret_1h',
            'nxt1_ret_2h', 'nxt1_ret_3h', 'nxt1_ret_5h', 'nxt1_ret_10h',
            'nxt1_ret_15h'
        ]
    ]

    total_data = total_factors.merge(total_return, on=['trade_time', 'code'])
    total_data = total_data[['trade_time', 'code', ret_name] + factors_cols]
    total_data.rename(columns={ret_name: 'nxt1_ret'}, inplace=True)
    return total_data, factors_cols


def df_to_tensors(total_data, factors_cols, device):
    """将 DataFrame 转换为 StackVM 所需的张量格式。

    Parameters
    ----------
    total_data : pd.DataFrame
        含 trade_time, code, nxt1_ret, 及 factors_cols 的数据。
    factors_cols : list[str]
        特征列名。
    device : torch.device

    Returns
    -------
    feat_tensor : torch.Tensor
        [N_assets, n_features, T_steps]
    target_ret : torch.Tensor
        [N_assets, T_steps]
    """
    import time
    t0 = time.time()

    # 获取所有的 unique 资产和时间，并确保它们是有序的
    assets = sorted(total_data['code'].unique())
    times = sorted(total_data['trade_time'].unique())

    # 将数据按照 [trade_time, code] 排序，以便后续 reshape
    total_data = total_data.sort_values(['trade_time', 'code'])

    # 建立多级索引，确保所有 (trade_time, code) 组合都存在。
    # 缺失的组合会被填充为 NaN
    multi_index = pd.MultiIndex.from_product([times, assets],
                                             names=['trade_time', 'code'])
    total_data = total_data.set_index(['trade_time',
                                       'code']).reindex(multi_index)

    n_assets = len(assets)
    n_times = len(times)
    n_features = len(factors_cols)

    # 抽取特征矩阵，形状转为 [T_steps, N_assets, n_features]
    feat_np = total_data[factors_cols].values.reshape(n_times, n_assets,
                                                      n_features)

    # 将其转置为所需的形状: [N_assets, n_features, T_steps]
    # np.transpose(feat_np, (1, 2, 0)) 的意思是：
    # 原维度 0(T_steps) 放到最后
    # 原维度 1(N_assets) 放到最前
    # 原维度 2(n_features) 放到中间
    feat_np = np.transpose(feat_np, (1, 2, 0)).astype(np.float32)

    # 抽取目标收益矩阵，形状转为 [T_steps, N_assets]
    ret_np = total_data['nxt1_ret'].values.reshape(n_times, n_assets)
    # 转置为 [N_assets, T_steps]
    ret_np = ret_np.T.astype(np.float32)

    # 处理 NaN
    feat_np = np.nan_to_num(feat_np, nan=0.0)
    ret_np = np.nan_to_num(ret_np, nan=0.0)

    feat_tensor = torch.tensor(feat_np, dtype=torch.float32, device=device)
    target_ret = torch.tensor(ret_np, dtype=torch.float32, device=device)

    t1 = time.time()
    print(f"向量化转换数据耗时: {t1-t0:.2f} 秒")

    return feat_tensor, target_ret


def callback_models(discoveries, custom_params):

    def create_params(params):
        m = hashlib.md5()
        # params可能是字典类型，需要转换为字符串
        if isinstance(params, dict):
            # 将字典按键排序后转换为字符串，确保相同参数组合产生相同hash
            params_str = str(sorted(params.items()))
        else:
            params_str = str(params)
        m.update(bytes(params_str, encoding='UTF-8'))
        return create_id(original=m.hexdigest(), digit=16)

    best_programs = pd.DataFrame(discoveries)
    
    task_id = str(custom_params['task_id'])
    session = str(custom_params['session'])
    tournament_size = int(custom_params['tournament_size'])
    standard_score = float(custom_params['standard_score'])
    best_programs['features'] = best_programs['expression'].apply(
        lambda x: create_params(x))
    best_programs.rename(columns={
        'expression': 'forumla',
        'score': 'final_fitness',
        'step': 'gen'
    },
                         inplace=True)

    dirs = os.path.join(base_path, custom_params['method'], "miner",
                        str(custom_params['task_id']),
                        custom_params['ret_name'],
                        str(custom_params['session']))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    programs_filename = os.path.join(dirs,
                                     f'programs_{task_id}_{session}.feather')

    logger.info(f"{programs_filename}")
    if os.path.exists(programs_filename):
        old_programs = pd.read_feather(programs_filename)
        best_programs = pd.concat([old_programs, best_programs], axis=0)

    final_programs = best_programs[
        (best_programs['final_fitness'] > standard_score)
        & (best_programs['final_fitness'] > 0)]

    final_programs = final_programs.drop_duplicates(subset=['features'])

    if final_programs.shape[0] > tournament_size:
        final_programs = final_programs.sort_values('final_fitness',
                                                    ascending=False)
        final_programs = final_programs.head(tournament_size)
    final_programs = final_programs.sort_values('final_fitness',
                                                ascending=False)
    logger.table(data=final_programs.head(5), title="final programs")
    final_programs.reset_index(drop=True).to_feather(programs_filename)


def run(factors_cols, feat_tensor, target_ret, callback, custom_params):
    # ---- 3. 构建模型 (特征外部传入) ----
    model = AlphaGPT(features_list=factors_cols)
    model.to(ModelConfig.DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"词汇表: {len(factors_cols)} features + "
          f"{model.vocab_size - len(factors_cols)} ops = "
          f"{model.vocab_size} tokens")

    # ---- 4. 构建引擎 ----
    engine = AlphaEngine(
        model=model,
        feat_tensor=feat_tensor,
        target_ret=target_ret,
        features_list=factors_cols,
    )

    # ---- 5. 训练 ----
    discoveries = engine.train(
        train_steps=ModelConfig.TRAIN_STEPS,
        batch_size=ModelConfig.BATCH_SIZE,
    )
    callback(discoveries, custom_params)


# ── 训练主函数 ────────────────────────────────────────────
def train(method, task_id, session, ret_name):
    # ---- 1. 加载数据 ----
    total_data, factors_cols = load_data(method=method,
                                         task_id=task_id,
                                         ret_name=ret_name)

    logger.info(f"数据加载完成: {total_data.shape}, 特征数: {len(factors_cols)}")
    # ---- 2. 转换为张量 ----
    ## 数据做了对齐，每期数据都是一样，382个标的 若没上市 这用0填充 所以是 382个标的 * 184个特征 * 20439时间
    feat_tensor, target_ret = df_to_tensors(total_data, factors_cols,
                                            ModelConfig.DEVICE)
    logger.info(f"张量构建完成: feat={feat_tensor.shape}, ret={target_ret.shape}")

    custom_params = {}
    custom_params['tournament_size'] = 600
    custom_params['standard_score'] = 0.01
    custom_params['method'] = method
    custom_params['task_id'] = task_id
    custom_params['session'] = session
    custom_params['ret_name'] = ret_name
    i = 0
    while i < 20:
        run(factors_cols=factors_cols,
            feat_tensor=feat_tensor,
            target_ret=target_ret,
            custom_params=custom_params,
            callback=callback_models)
        i += 1


if __name__ == '__main__':
    variant = Tactix().start()
    train(method=variant.method,
          task_id=variant.task_id,
          session=variant.session,
          ret_name=variant.ret_name)
