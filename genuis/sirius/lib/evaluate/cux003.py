## 新增固定方向 (支持 auto 控制是否在 IC<0 时反转因子方向)
import os
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from lib.evaluate.cux001 import FactorEvaluate1 as FactorEvaluate1001
except ImportError:
    from .cux001 import FactorEvaluate1 as FactorEvaluate1001


class FactorEvaluate1(FactorEvaluate1001):
    """时间序列单因子评估器 (支持 auto=False 固定方向与 auto=True 自动反转方向)."""

    def __init__(self,
                 factor_data,
                 resampling_win: int = 1,
                 factor_name: str = 'factor',
                 ret_name: str = 'ret',
                 roll_win: int = 252,
                 fee: float = 0.0003,
                 scale_method: str = 'roll_min_max',
                 annualization_factor: int = 252,
                 expression=None,
                 name=None,
                 auto: bool = False):
        
        name1 = name if isinstance(name, str) else factor_name
        name = name1 + "_" + str(resampling_win) + "h"
        super().__init__(
            factor_data=factor_data,
            resampling_win=resampling_win,
            factor_name=factor_name,
            ret_name=ret_name,
            roll_win=roll_win,
            fee=fee,
            scale_method=scale_method,
            annualization_factor=annualization_factor,
            expression=expression if isinstance(expression, str) else factor_name,
            name=name
        )
        self.auto = auto

    def run(self, is_check: bool = False) -> dict:
        """执行因子评估并输出回测与 IC 统计指标."""
        stats = super().run(is_check=is_check)
        if is_check:
            if getattr(self, 'direction_inverted', False):
                print("INFO: IC Mean is negative. Factor direction has been inverted (auto=True).")
            else:
                print(f"INFO: Factor direction kept unchanged (auto={self.auto}).")
        return stats
