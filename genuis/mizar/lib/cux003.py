import pdb
import os, hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  #
import seaborn as sns

from lib.cux001 import FactorEvaluate1 as FactorEvaluate1001


class FactorEvaluate1(FactorEvaluate1001):

    def __init__(self,
                 factor_data: pd.DataFrame,
                 resampling_win: int = 1,
                 factor_name: str = 'factor',
                 ret_name: str = 'ret',
                 roll_win: int = 252,
                 fee: float = 0.0003,
                 scale_method: str = 'roll_min_max',
                 annualization_factor: int = 252,
                 expression=None,
                 name=None,
                 auto=False):

        super(FactorEvaluate1,
              self).__init__(factor_data=factor_data,
                             resampling_win=resampling_win,
                             factor_name=factor_name,
                             ret_name=ret_name,
                             roll_win=roll_win,
                             fee=fee,
                             scale_method=scale_method,
                             annualization_factor=annualization_factor,
                             expression=expression,
                             name=name)
        self.auto = auto

    def run(self, is_check=False):
        ### 滚动标准化
        self._scale()
        ### 重采样
        if self.resampling_win <= 1:
            print("WARINING: resampling_win:{0}".format(self.resampling_win))
        is_on_mark = self.factor_data.index.get_level_values(
            level=0).minute % int(self.resampling_win) == 0
        self.resample_data = self.factor_data[is_on_mark].copy()

        ic_stats = self.cal_ic()
        if ic_stats['ic_mean'] < 0 and self.auto: ## 新增一个是否自动控制方向
            self.resample_data['f_scaled'] *= -1
            #ic_stats = self.cal_ic()
            if is_check:
                print("INFO: IC Mean is negative. Factor has been inverted.")
        if self.resample_data['f_scaled'].dropna().empty:
            return {
                'total_ret': -1.0,
                'avg_ret': -1.0,
                'calmar': -10.0,
                'sharpe1': -1.0,
                'sharpe2': -10.0,
                'turnover': 10.0,
                'win_rate': 0,
                'profit_ratio': 0.0,
                'total_ic': 0.0,
                'ic_mean': 0.00,
                'ic_std': 1.0,
                'ic_ir': 1.0,
                'factor_autocorr': 1.0,
                'ret_autocorr': 1.0
            }
        pnl_stats = self.cal_pnl()
        autocorr_stats = self._cal_autocorr()  # 计算自相关性

        # 合并所有统计数据
        pnl_stats.update(ic_stats)
        pnl_stats.update(autocorr_stats)

        self.stats = pnl_stats
        if is_check:
            self._check_warnings()  # 运行警告检查
        return self.stats
