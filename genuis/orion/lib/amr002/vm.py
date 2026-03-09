# -*- coding: utf-8 -*-
"""
StackVM — 基于栈的公式执行虚拟机。

复用 model_core/vm.py 的 torch 张量算子模式:
  - token < n_features  → 取特征切片 feat_tensor[:, token, :]
  - token >= n_features → 从 OPS_CONFIG 中取算子执行

与 model_core 的区别: n_features 由外部特征数量决定, 不再硬编码。
"""
import torch
from .ops import OPS_CONFIG


class StackVM:
    """基于栈的因子公式执行虚拟机。

    Parameters
    ----------
    n_features : int
        外部特征的数量, 即词汇表中特征 token 的数量。
    """

    def __init__(self, n_features):
        self.n_features = n_features
        # 算子 token 从 n_features 开始编号
        self.op_map = {
            i + n_features: cfg[1] for i, cfg in enumerate(OPS_CONFIG)
        }
        self.arity_map = {
            i + n_features: cfg[2] for i, cfg in enumerate(OPS_CONFIG)
        }

    def execute(self, formula_tokens, feat_tensor):
        """执行公式, 返回因子值张量。

        Parameters
        ----------
        formula_tokens : list[int]
            token id 序列。
        feat_tensor : torch.Tensor
            形状 [N_assets, n_features, T_steps] 的特征张量。

        Returns
        -------
        torch.Tensor or None
            形状 [N_assets, T_steps] 的因子值; 无效公式返回 None。
        """
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.n_features:
                    # 特征节点: 取对应特征切片
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity:
                        return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(
                            res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            if len(stack) == 1:
                return stack[0]
            else:
                return None
        except Exception:
            return None

    def decode_readable(self, formula_tokens, features_list):
        """将 token id 序列解码为人类可读的公式字符串。

        Parameters
        ----------
        formula_tokens : list[int]
            token id 序列。
        features_list : list[str]
            特征名列表。

        Returns
        -------
        str or None
        """
        ops_names = {
            i + self.n_features: cfg[0] for i, cfg in enumerate(OPS_CONFIG)
        }
        ops_arity = self.arity_map

        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.n_features:
                    stack.append(f"'{features_list[token]}'")
                elif token in ops_names:
                    arity = ops_arity[token]
                    if len(stack) < arity:
                        return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    name = ops_names[token]
                    
                    # 尝试解析带周期的时间序列算子 (例: EMA_10 -> EMA(10, arg))
                    if '_' in name:
                        base_name, period = name.rsplit('_', 1)
                        if period.isdigit():
                            expr = f"{base_name}({period}, {', '.join(args)})"
                        else:
                            expr = f"{name}({', '.join(args)})"
                    else:
                        expr = f"{name}({', '.join(args)})"
                        
                    stack.append(expr)
                else:
                    return None
            return stack[0] if len(stack) == 1 else None
        except Exception:
            return None
