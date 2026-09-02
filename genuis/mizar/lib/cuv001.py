import pdb
import os, hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  #
import seaborn as sns
from xml.dom import minidom
from xml.etree import ElementTree as ET
'''
scale_method: str = 'roll_min_max':
作用: 指定对因子进行放缩的方法。这是一个字符串，它有以下几种可选值，并且都严格遵守“无未来数据”原则：
'roll_min_max': 滚动窗口内的 Min-Max 放缩，将因子值放缩到 [-1, 1] 之间。
'roll_zscore': 滚动窗口内的 Z-score 放缩，将因子值转换为标准差单位，并截断（clip）到 [-3, 3] 后归一化到 [-1, 1]。
'roll_quantile': 滚动窗口内的分位数放缩，使用 25% 和 75% 分位数（四分位距）进行放缩，也映射到 [-1, 1]，对极端值更鲁棒。
'ew_zscore': 基于指数加权移动平均 (EWM) 的 Z-score 放缩，对近期数据赋予更高权重。
'train_const': 使用前 roll_win 个样本的均值和标准差作为固定参数来放缩整个时间序列的因子。这意味着在初始窗口之后，放缩参数是常量，不再滚动变化。
'''


def generate_simple_id(formula: str) -> str:
    # 1. 移除空格并转为小写
    normalized_formula = formula.replace(" ", "").lower()

    # 2. 使用 MD5 哈希
    # .encode('utf-8') 是必须的，因为哈希函数处理的是字节
    hasher = hashlib.md5(normalized_formula.encode('utf-8'))

    # .hexdigest() 返回16进制的哈希字符串
    return hasher.hexdigest()


# 建议设置一个美观的绘图风格
sns.set_style('whitegrid')


class FactorEvaluate1(object):

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
                 name=None):

        self.factor_data = factor_data.copy()
        self.factor_name = factor_name
        self.ret_name = ret_name
        self.roll_win = roll_win
        self.fee = fee
        self.scale_method = scale_method
        self.annualization_factor = annualization_factor
        self.name = name
        self.expression = expression
        self.stats = None
        self.resampling_win = int(resampling_win)
        self._init_factor()

    def _init_factor(self):
        self.factor_data['trade_time'] = pd.to_datetime(
            self.factor_data['trade_time'])
        self.factor_data.set_index('trade_time', inplace=True)
        self.factor_data = self.factor_data.sort_index()[[
            self.factor_name, self.ret_name
        ]]

    def _scale(self):
        x = self.factor_data[self.factor_name]
        win = self.roll_win
        #roll_min_max:
        #rmin = x.rolling(win).min(): 计算滚动窗口内的最小值。
        #rmax = x.rolling(win).max(): 计算滚动窗口内的最大值。
        #self.factor_df['f_scaled'] = 2 * (x - rmin) / (rmax - rmin).clip(lower=1e-8) - 1: 将因子值线性映射到 [-1, 1] 之间。.clip(lower=1e-8) 是为了防止分母为零而导致计算错误。
        if self.scale_method == 'roll_min_max':
            rmin = x.rolling(win).min()
            rmax = x.rolling(win).max()
            self.factor_data['f_scaled'] = 2 * \
                (x - rmin) / (rmax - rmin).clip(lower=1e-8) - 1

        # mu = x.rolling(win).mean(): 计算滚动窗口内的均值。
        # sg = x.rolling(win).std(): 计算滚动窗口内的标准差。
        # self.factor_df['f_scaled'] = ((x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3: 计算滚动 Z-score，并将其值截断到 [-3, 3] 范围内，然后除以3，使其结果也归一化到 [-1, 1]。
        elif self.scale_method == 'roll_zscore':
            mu = x.rolling(win).mean()
            sg = x.rolling(win).std()
            self.factor_data['f_scaled'] = (
                (x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3

        # roll_quantile:
        # q25 = x.rolling(win).quantile(0.25): 计算滚动窗口内的 25% 分位数。
        # q75 = x.rolling(win).quantile(0.75): 计算滚动窗口内的 75% 分位数。
        # self.factor_df['f_scaled'] = 2 * (x - q25) / (q75 - q25).clip(lower=1e-8) - 1: 基于四分位距进行放缩，映射到 [-1, 1]。
        elif self.scale_method == 'roll_quantile':
            q25 = x.rolling(win).quantile(0.25)
            q75 = x.rolling(win).quantile(0.75)
            self.factor_data['f_scaled'] = 2 * \
                (x - q25) / (q75 - q25).clip(lower=1e-8) - 1

        #ew_zscore:
        #ema = x.ewm(span=win, adjust=False).mean(): 计算指数加权移动平均。
        #evar = x.ewm(span=win, adjust=False).var(): 计算指数加权移动方差。
        #self.factor_df['f_scaled'] = ((x - ema) / np.sqrt(evar).clip(lower=1e-8)).clip(-3, 3) / 3: 基于指数加权统计量计算 Z-score，并进行截断和归一化。
        elif self.scale_method == 'ew_zscore':
            ema = x.ewm(span=win, adjust=False).mean()
            evar = x.ewm(span=win, adjust=False).var()
            self.factor_data['f_scaled'] = (
                (x - ema) / np.sqrt(evar).clip(lower=1e-8)).clip(-3, 3) / 3

        #train_const:
        #mu = x.iloc[:win].mean(): 使用前 win 个（即训练集）样本的均值。
        #sg = x.iloc[:win].std(): 使用前 win 个样本的标准差。
        #self.factor_df['f_scaled'] = ((x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3: 使用固定的均值和标准差对所有因子值进行 Z-score 放缩，然后截断和归一化。
        elif self.scale_method == 'train_const':
            # 用前 roll 个样本做训练集
            mu = x.iloc[:win].mean()
            sg = x.iloc[:win].std()
            self.factor_data['f_scaled'] = (
                (x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3

        elif self.scale_method == 'raw':
            # 直接使用原始值，不进行任何缩放，假设为已经处理好的因子值，离散值为[-1,0,1], 连续值为[-1，1]
            self.factor_data['f_scaled'] = x
        else:
            raise ValueError('Unknown scale_method')

    def cal_ic(self):
        """
        计算因子与预期收益的滚动相关性
        """
        self.resample_data['ic'] = self.resample_data[self.ret_name].rolling(
            window=self.roll_win,
            min_periods=5).corr(self.resample_data[self.factor_name])
        total_ic = self.resample_data[self.ret_name].corr(
            self.resample_data[self.factor_name])
        self.resample_data['cumsum_ic'] = self.resample_data['ic'].cumsum()
        ic_mean = self.resample_data['ic'].mean()
        ic_std = self.resample_data['ic'].std()
        return {
            'total_ic': total_ic,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_mean / ic_std if ic_std != 0 else 0  # 衡量因子预测能力的稳定性和质量。
        }

    def cal_pnl(self):
        # 缺失信号按目标空仓处理，同时会正确计入平仓换手。
        self.resample_data['pos'] = self.resample_data['f_scaled'].fillna(0.0)

        self.resample_data[
            'gross_ret'] = self.resample_data['pos'] * self.resample_data[
                self.ret_name]  # 计算每期的总收益（未扣除费用），即头寸乘以对应期的远期收益。

        self.resample_data['turnover'] = self.resample_data['pos'].diff().abs()
        self.resample_data.loc[self.resample_data.index[0], 'turnover'] = abs(
            self.resample_data['pos'].iloc[0])

        self.resample_data['net_ret'] = (
            self.resample_data['gross_ret'] -
            self.fee * self.resample_data['turnover']
        )  # 计算每期的净收益，即从总收益中减去交易费用。费用是换手率乘以设定的 fee。

        self.resample_data['nav'] = (
            1 + self.resample_data['net_ret']
        ).cumprod(
        )  #  计算净值曲线（Net Asset Value）。这是 (1 + 净收益) 的累积乘积，代表了投资组合的模拟价值随时间的变化。

        ## 1. 计算回测跨越的年数
        delta_days = (self.resample_data.index[-1] -
                      self.resample_data.index[0]).days
        years = delta_days / 365
        if years <= 0: years = 1e-6  # 防止除零

        # -------- 基础统计 --------
        total_ret = self.resample_data['nav'].iloc[-1] - 1  # 累计收益 整个回测期间的累计收益。

        avg_ret = self.resample_data['net_ret'].mean()  # 平均每次交易收益

        ann_ret = (1 + total_ret)**(1 / years) - 1  # 计算年化

        running_max = self.resample_data['nav'].cummax().clip(lower=1.0)
        max_dd = (self.resample_data['nav'] / running_max - 1).min()

        calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan  # 卡玛比率

        ## 换算日收益率 算夏普
        daily_net_ret = (1 + self.resample_data['net_ret']).groupby(
            self.resample_data.index.normalize()).prod() - 1

        rets_mean = daily_net_ret.mean() * self.annualization_factor
        rets_std = daily_net_ret.std() * np.sqrt(self.annualization_factor)
        sharpe2 = (rets_mean / rets_std
                   if np.isfinite(rets_std) and rets_std > 0 else 0)
        period_std = self.resample_data['net_ret'].std()
        sharpe1 = (self.resample_data['net_ret'].mean() / period_std
                   if np.isfinite(period_std) and period_std > 0 else 0)

        turnover = self.resample_data['turnover'].mean()  # 平均每期换手率。

        win_rate = (self.resample_data['net_ret']
                    > 0).mean()  # 胜率，即净收益为正的周期所占的比例。

        winning_returns = self.resample_data.loc[
            self.resample_data['net_ret'] > 0, 'net_ret']
        losing_returns = self.resample_data.loc[
            self.resample_data['net_ret'] < 0, 'net_ret']
        profit_ratio = (winning_returns.mean() / abs(losing_returns.mean())
                        if not losing_returns.empty else np.inf)
        if winning_returns.empty:
            profit_ratio = 0.0

        return {
            'total_ret': total_ret,
            'avg_ret': avg_ret,
            'max_dd': max_dd,
            'calmar': calmar,
            'sharpe1': sharpe1,
            'sharpe2': sharpe2,
            'turnover': turnover,
            'win_rate': win_rate,
            'profit_ratio': profit_ratio
        }

    def cal_returns(self):
        """计算多空收益情况"""
        direction = np.sign(self.resample_data['f_scaled'].values)
        long_returns = self.resample_data['net_ret'][direction > 0]
        short_returns = self.resample_data['net_ret'][direction < 0]
        long_sum_returns = long_avg_returns = long_win_ratio = 0.0
        short_sum_returns = short_avg_returns = short_win_ratio = 0.0

        long_count = len(long_returns)
        if long_count > 0:
            long_sum_returns = long_returns.sum()
            long_avg_returns = long_returns.mean()
            long_win_ratio = (long_returns > 0).mean()

        short_count = len(short_returns)
        if short_count > 0:
            short_sum_returns = short_returns.sum()
            short_avg_returns = short_returns.mean()
            short_win_ratio = (short_returns > 0).mean()

        return {
            "long_count": long_count,
            "long_sum_returns": long_sum_returns,
            "long_avg_returns": long_avg_returns,
            "long_win_ratio": long_win_ratio,
            "short_count": short_count,
            "short_sum_returns": short_sum_returns,
            "short_avg_returns": short_avg_returns,
            "short_win_ratio": short_win_ratio
        }

    def _cal_autocorr(self):
        """计算因子和收益率的滞后1期自相关性。"""
        factor_ac = self.resample_data[self.factor_name].autocorr(lag=1)
        ret_ac = self.resample_data[self.ret_name].autocorr(lag=1)
        return {'factor_autocorr': factor_ac, 'ret_autocorr': ret_ac}

    def _cal_distribution(self):
        factor = self.resample_data[self.factor_name]
        valid_factor = factor.dropna()
        coverage = factor.notna().mean()
        zero_rate = (valid_factor.abs().le(1e-12).mean()
                     if not valid_factor.empty else np.nan)
        mean = factor.mean()
        std = factor.std()
        skew = factor.skew()
        kurtosis = factor.kurt()
        return {
            'factor_coverage': coverage,
            'factor_zero_rate': zero_rate,
            'factor_mean': mean,
            'factor_std': std,
            'factor_skew': skew,
            'factor_kurtosis': kurtosis
        }

    def _check_warnings(self):
        """检查关键指标并打印警告。"""
        print("\n--- Sanity Checks & Warnings ---")
        # 1. 检查收益率自相关性
        ret_ac = self.stats.get('ret_autocorr', 0)
        if not (-0.1 < ret_ac < 0.1):
            print(
                f"⚠️  WARNING: Return autocorrelation is {ret_ac:.3f}. Normal range is [-0.1, 0.1]. High value might indicate data issues (e.g., stale prices)."
            )
        else:
            print(f"✅ Return autocorrelation ({ret_ac:.3f}) is normal.")

        # 2. 检查因子自相关性 (极端值)
        factor_ac = self.stats.get('factor_autocorr', 0)
        if abs(factor_ac) > 0.99:
            print(
                f"⚠️  WARNING: Factor autocorrelation is {factor_ac:.3f}, which is extremely high. The factor is nearly non-stationary and may have high turnover."
            )
        else:
            print(
                f"✅ Factor autocorrelation ({factor_ac:.3f}) is within a reasonable range."
            )

        # 3. 检查ICIR
        ic_ir = self.stats.get('ic_ir', 0)
        if ic_ir < 0.3:
            print(
                f"⚠️  WARNING: ICIR is {ic_ir:.3f}, which is low. Factor's predictive power is unstable."
            )
        else:
            print(f"✅ ICIR ({ic_ir:.3f}) indicates stable performance.")

    def run(self, is_check=False):
        ### 滚动标准化
        self._scale()
        ### 重采样
        if self.resampling_win <= 1:
            print("WARINING: resampling_win:{0}".format(self.resampling_win))
        is_on_mark = self.factor_data.index.get_level_values(
            level=0).minute % int(self.resampling_win) == 0
        self.resample_data = self.factor_data[is_on_mark].copy()

        self.resample_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.resample_data = self.resample_data.loc[
            self.resample_data[self.ret_name].notna()].copy()
        if self.resample_data.empty:
            raise ValueError('No valid return observations after resampling.')

        ic_stats = self.cal_ic()
        self.direction_inverted = bool(ic_stats['ic_mean'] < 0)
        if self.direction_inverted:
            self.resample_data['f_scaled'] *= -1
            #ic_stats = self.cal_ic()
            if is_check:
                print("INFO: IC Mean is negative. Factor has been inverted.")
        if self.resample_data['f_scaled'].dropna().empty:
            self.stats = {
                'total_ret': -1.0,
                'avg_ret': -1.0,
                'max_dd': np.nan,
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
                'ret_autocorr': 1.0,
                'factor_coverage': 0.0,
                'factor_zero_rate': np.nan,
                'factor_mean': np.nan,
                'factor_std': np.nan,
                'factor_skew': np.nan,
                'factor_kurtosis': np.nan,
                'direction_inverted': self.direction_inverted,
                'effective_total_ic': np.nan,
                'effective_ic_mean': np.nan,
                'effective_ic_std': np.nan,
                'effective_ic_ir': np.nan,
            }
            return self.stats
        pnl_stats = self.cal_pnl()
        autocorr_stats = self._cal_autocorr()  # 计算自相关性
        distribution_stats = self._cal_distribution()  # 因子分布

        # 合并所有统计数据
        pnl_stats.update(ic_stats)
        pnl_stats.update(autocorr_stats)
        pnl_stats.update(distribution_stats)
        direction = -1.0 if self.direction_inverted else 1.0
        pnl_stats.update({
            'direction_inverted': self.direction_inverted,
            'effective_total_ic': ic_stats['total_ic'] * direction,
            'effective_ic_mean': ic_stats['ic_mean'] * direction,
            'effective_ic_std': ic_stats['ic_std'],
            'effective_ic_ir': ic_stats['ic_ir'] * direction,
        })

        self.stats = pnl_stats
        if is_check:
            self._check_warnings()  # 运行警告检查
        return self.stats

    def _generate_stats_text(self) -> str:
        """生成性能统计摘要文本"""
        report_parts = []
        if self.expression is not None:
            report_parts.append(f"Expression: {self.expression}")
        if self.name is not None:
            report_parts.append(f"Name: {self.name}")

        if len(report_parts) > 0:
            report_parts.append("\n")

        performance_metrics = (
            f"--- Performance Metrics ---\n"
            f"{'Avg Return (bps)':<25}: {(self.stats.get('avg_ret', float('nan')) * 10000):.2f}\n"
            f"{'Total Return':<25}: {self.stats['total_ret']:.2%}\n"
            f"{'Sharpe Ratio':<25}: {self.stats['sharpe1']:.2f}\n"
            f"{'Ann Sharpe Ratio':<25}: {self.stats['sharpe2']:.2f}\n"
            f"{'Max Drawdown':<25}: {self.stats['max_dd']:.2%}\n"
            f"{'Calmar Ratio':<25}: {self.stats.get('calmar', float('nan')):.2f}\n"
            f"{'Win Rate':<25}: {self.stats['win_rate']:.2%}\n"
            f"{'Profit/Loss Ratio':<25}: {self.stats['profit_ratio']:.2f}\n"
            f"\n--- Factor Characteristics ---\n"
            f"{'IC Mean':<25}: {self.stats['ic_mean']:.4f}\n"
            f"{'ICIR':<25}: {self.stats['ic_ir']:.4f}\n"
            f"{'Mean Turnover':<25}: {self.stats['turnover']:.4f}\n"
            f"{'Factor Coverage':<25}: {self.stats['factor_coverage']:.2%}\n"
            f"{'Factor Zero Rate':<25}: {self.stats['factor_zero_rate']:.2%}\n"
            f"{'Factor Autocorr':<25}: {self.stats['factor_autocorr']:.4f}\n"
            f"{'Return Autocorr':<25}: {self.stats['ret_autocorr']:.4f}\n")
        report_parts.append(performance_metrics)
        return "\n".join(report_parts)

    def plot_results(self,
                     max_line_points: int = 20000,
                     max_scatter_points: int = 30000,
                     scatter_random_state: int = 42):
        """Plot evaluation results with display-only downsampling.

        Downsampling never changes ``resample_data`` or any reported metric.
        Pass ``None`` for either point limit to disable that downsampling.
        """
        if self.stats is None:
            raise RuntimeError(
                "Please run the 'run()' method before plotting.")

        def downsample_series(series, max_points):
            """Evenly sample a series while always retaining both endpoints."""
            if max_points is None or len(series) <= max_points:
                return series
            if max_points < 2:
                raise ValueError('max_line_points must be at least 2 or None')
            positions = np.linspace(
                0, len(series) - 1, num=max_points, dtype=np.int64)
            return series.iloc[np.unique(positions)]

        def set_sequential_xticks(ax, series, num_ticks=7):
            """
            为一个使用整数索引绘图的坐标轴设置日期标签。
            ax: a matplotlib axis object.
            series: The original pandas Series with a DatetimeIndex.
            num_ticks: The desired number of date labels on the x-axis.
            """
            # 计算刻度的整数位置
            tick_positions = np.linspace(0,
                                         len(series) - 1,
                                         num_ticks,
                                         dtype=int)
            # 获取这些位置对应的日期标签
            tick_labels = [
                series.index[i].strftime('%Y-%m-%d') for i in tick_positions
            ]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=30, ha='right')

        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle(
            f"Factor Evaluation: {self.factor_name} vs {self.ret_name} | roll_win={self.roll_win}, resampling_win={self.resampling_win}, scale_method={self.scale_method}",
            fontsize=18)

        # 1. 净值曲线 (NAV)
        ax1 = axes[0, 0]
        nav_data = downsample_series(
            self.resample_data['nav'].dropna(), max_line_points)
        gross_ret_data = downsample_series(
            (1 + self.resample_data['gross_ret']).cumprod().dropna(),
            max_line_points)

        # 使用 use_index=False 来忽略时间轴，绘制连续序列
        nav_data.plot(ax=ax1,
                      label='Net Asset Value (NAV)',
                      color='blue',
                      use_index=False)
        gross_ret_data.plot(ax=ax1,
                            label='Cumulative Gross Return',
                            color='orange',
                            linestyle='--',
                            use_index=False)

        # 使用辅助函数设置X轴标签
        set_sequential_xticks(ax1, nav_data)

        ax1.set_title("Performance")
        ax1.set_ylabel("NAV")
        ax1.set_xlabel("trade_time (sequential)")  # 标签提示X轴是序列
        ax1.legend()
        ax1.grid(True)

        # 2. 绩效指标表格
        ax_table = axes[0, 1]
        ax_table.axis('off')

        report_parts = []
        if self.expression is not None:
            report_parts.append(f"Expression: {self.expression}")
        if self.name is not None:
            report_parts.append(f"Name: {self.name}")

        if len(report_parts) > 0:
            report_parts.append("\n")

        performance_metrics = (
            f"--- Performance Metrics ---\n"
            f"{'Avg Return':<20}: {(self.stats.get('avg_ret', float('nan')) * 10000):.2f} bps\n"
            f"{'Total Return':<20}: {self.stats['total_ret']:.2%}\n"
            f"{'Sharpe Ratio':<20}: {self.stats['sharpe1']:.2f}\n"
            f"{'Ann Sharpe Ratio':<20}: {self.stats['sharpe2']:.2f}\n"
            f"{'Max Drawdown':<20}: {self.stats['max_dd']:.2%}\n"
            f"{'Calmar Ratio':<20}: {self.stats.get('calmar', float('nan')):.2f}\n"
            f"{'Win Rate':<20}: {self.stats['win_rate']:.2%}\n"
            f"{'Profit/Loss Ratio':<20}: {self.stats['profit_ratio']:.2f}\n"
            f"\n--- Factor Characteristics ---\n"
            f"{'Total IC':<20}: {self.stats['total_ic']:.4f}\n"
            f"{'IC Mean':<20}: {self.stats['ic_mean']:.4f}\n"
            f"{'ICIR':<20}: {self.stats['ic_ir']:.4f}\n"
            f"{'Mean Turnover':<20}: {self.stats['turnover']:.4f}\n"
            f"{'Factor Coverage':<20}: {self.stats['factor_coverage']:.2%}\n"
            f"{'Factor Zero Rate':<20}: {self.stats['factor_zero_rate']:.2%}\n"
            f"{'Factor Autocorr':<20}: {self.stats['factor_autocorr']:.4f}\n"  # 新增
            f"{'Return Autocorr':<20}: {self.stats['ret_autocorr']:.4f}\n"  # 新增
            f"{'Roll Window':<20}: {self.roll_win}\n"
            f"{'Resampling Window':<20}: {self.resampling_win}\n")
        report_parts.append(performance_metrics)
        stats_text = "\n".join(report_parts)

        ax_table.text(0.05,
                      0.95,
                      stats_text,
                      transform=ax_table.transAxes,
                      fontsize=12,
                      verticalalignment='top',
                      fontfamily='monospace',
                      linespacing=1.1)
        ax_table.set_title("Key Performance Indicators", fontsize=14)

        # 3. IC 和 累计IC
        ax3 = axes[1, 0]
        ic_data = downsample_series(
            self.resample_data['ic'].dropna(), max_line_points)
        cumsum_ic_data = downsample_series(
            self.resample_data['cumsum_ic'].dropna(), max_line_points)

        ic_data.plot(ax=ax3,
                     label='Rolling IC',
                     color='steelblue',
                     alpha=0.8,
                     use_index=False)
        set_sequential_xticks(ax3, ic_data)

        ax3.set_ylabel("Rolling IC", color='steelblue')
        ax3_twin = ax3.twinx()

        cumsum_ic_data.plot(ax=ax3_twin,
                            label='Cumulative IC',
                            color='black',
                            linestyle='--',
                            use_index=False)

        ax3_twin.set_ylabel("Cumulative IC", color='black')
        ax3.set_title("IC Analysis")
        ax3.set_xlabel("trade_time (sequential)")
        ax3.axhline(0, color='red', linestyle='--', linewidth=1)
        ax3.grid(True)

        # 4. 因子 vs. 收益率散点图
        ax4 = axes[1, 1]
        scatter_data = self.resample_data[
            [self.factor_name, self.ret_name]].dropna()
        if (max_scatter_points is not None
                and len(scatter_data) > max_scatter_points):
            if max_scatter_points < 1:
                raise ValueError(
                    'max_scatter_points must be positive or None')
            scatter_data = scatter_data.sample(
                n=max_scatter_points, random_state=scatter_random_state)
        ax4.scatter(scatter_data[self.factor_name],
                    scatter_data[self.ret_name],
                    s=10,
                    alpha=0.3,
                    color='purple',
                    edgecolors='none',
                    rasterized=True)
        ax4.set_title("Factor vs. Return Scatter Plot")
        ax4.set_xlabel("Original Factor Value")
        ax4.set_ylabel("Forward Return")
        ax4.grid(True)

        # 5. 每日收益率与回撤
        ax5 = axes[2, 0]
        drawdown_data = downsample_series((
            (self.resample_data['nav'] / self.resample_data['nav'].cummax() -
             1) * 100).dropna(), max_line_points)

        drawdown_data.plot(ax=ax5, color='red', alpha=0.8, use_index=False)
        # fill_between 需要 numpy 数组
        ax5.fill_between(np.arange(len(drawdown_data)),
                         drawdown_data.values,
                         0,
                         color='red',
                         alpha=0.2)
        set_sequential_xticks(ax5, drawdown_data)

        ax5.set_title(f"Drawdown Over Time (Max = {self.stats['max_dd']:.2%})")
        ax5.set_ylabel("Drawdown (%)")
        ax5.set_xlabel("trade_time (sequential)")
        ax5.set_ylim(bottom=None, top=0.5)
        ax5.grid(True)

        # 6. 换手率时序图
        ax6 = axes[2, 1]
        turnover_data = downsample_series(
            self.resample_data['turnover'].dropna(), max_line_points)

        turnover_data.plot(ax=ax6, color='teal', use_index=False)
        set_sequential_xticks(ax6, turnover_data)

        ax6.set_title(
            f"Turnover Over Time (Mean = {self.stats['turnover']:.3f})")
        ax6.set_ylabel("Turnover")
        ax6.set_xlabel("trade_time (sequential)")
        ax6.grid(True)

        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter('%Y-%m')

        # 2. 将格式应用到所有需要时间轴的子图
        #for ax in [ax1, ax3, ax5, ax6]:
        #    ax.xaxis.set_major_locator(locator)
        #    ax.xaxis.set_major_formatter(formatter)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96], h_pad=4.0)
        plt.show()
        self.figure = fig

    def generate_xml(self, start_time, end_time):

        def add_text(parent, tag, value, **attributes):
            element = ET.SubElement(
                parent, tag, {
                    key: str(item)
                    for key, item in attributes.items() if item is not None
                })
            if value is None:
                element.set('status', 'missing')
            else:
                element.text = str(value)
            return element

        def add_metric(parent, tag, value, description, unit='ratio'):
            element = ET.SubElement(parent, tag, {
                'description': description,
                'unit': unit,
            })
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = np.nan
            if np.isfinite(numeric_value):
                element.text = format(numeric_value, '.12g')
            else:
                element.set('status', 'missing')
            return element

        root = ET.Element('factor_evaluation', {
            'schema_version': '1.0.0',
            'purpose': 'factor_screening',
        })

        identity = ET.SubElement(root, 'identity')
        add_text(identity, 'name', self.name)
        add_text(identity, 'expression', self.expression)
        add_text(identity, 'factor_field', self.factor_name)
        add_text(identity, 'forward_return_field', self.ret_name)

        evaluation = ET.SubElement(root, 'evaluation_config')
        add_text(evaluation, 'start_time', start_time)
        add_text(evaluation, 'end_time', end_time)
        add_text(evaluation, 'rolling_window', self.roll_win, unit='period')
        add_text(evaluation,
                 'resampling_window',
                 self.resampling_win,
                 unit='minute')

        add_text(evaluation, 'scale_method', self.scale_method)
        add_text(evaluation,
                 'annualization_factor',
                 self.annualization_factor,
                 unit='trading_days_per_year')
        add_text(evaluation, 'return_basis', 'net_return_after_costs')

        direction = ET.SubElement(root, 'direction_adjustment')
        add_text(direction,
                 'applied',
                 str(self.stats.get('direction_inverted', False)).lower(),
                 unit='boolean')
        add_text(
            direction, 'rule',
            'invert f_scaled when pre_adjustment rolling IC mean is negative')

        metrics = ET.SubElement(root, 'metrics', {
            'basis': 'net_return_after_costs',
        })
        add_metric(metrics, 'average_return', self.stats['avg_ret'], '平均收益',
                   'decimal_return')
        add_metric(metrics, 'total_return', self.stats['total_ret'], '累计成本后收益',
                   'decimal_return')
        add_metric(metrics, 'annualized_sharpe', self.stats['sharpe2'], '年化夏普',
                   'ratio')
        add_metric(metrics, 'maximum_drawdown', self.stats['max_dd'], '最大回撤',
                   'decimal_return')
        add_metric(metrics, 'calmar_ratio', self.stats['calmar'], '卡玛',
                   'ratio')
        add_metric(metrics, 'win_rate', self.stats['win_rate'], '胜率',
                   'decimal_ratio')
        add_metric(metrics, 'profit_loss_ratio', self.stats['profit_ratio'],
                   '盈亏比', 'ratio')
        add_metric(metrics, 'turnover', self.stats['turnover'], '平均换手率',
                   'turnover')

        characteristics = ET.SubElement(root, 'factor_characteristics', {
            'ic_scope': 'time_series',
        })
        effective_ic = ET.SubElement(characteristics, 'effective_ic', {
            'description': '方向调整后实际用于收益计算的因子IC',
        })
        add_metric(effective_ic, 'total_ic', self.stats['effective_total_ic'], '全样本IC',
                   'correlation')
        add_metric(effective_ic, 'ic_mean', self.stats['effective_ic_mean'], '滚动IC平均值',
                   'correlation')
        add_metric(effective_ic, 'ic_std', self.stats.get('effective_ic_std'), '滚动IC标准差',
                   'correlation')
        add_metric(effective_ic, 'ic_ir', self.stats.get('effective_ic_ir'), '滚动ICIR',
                   'ratio')

        add_metric(characteristics, 'factor_autocorrelation',
                   self.stats.get('factor_autocorr'), '原始因子一阶自相关',
                   'correlation')
        add_metric(characteristics, 'return_autocorrelation',
                   self.stats.get('ret_autocorr'), '目标收益一阶自相关', 'correlation')

        distribution = ET.SubElement(root, 'factor_distribution')
        add_metric(distribution, 'coverage', self.stats.get('factor_coverage'),
                   '有效因子值覆盖率', 'decimal_ratio')
        add_metric(distribution, 'zero_rate',
                   self.stats.get('factor_zero_rate'),
                   '有效原始因子中近似为零的比例', 'decimal_ratio')
        add_metric(distribution, 'mean', self.stats.get('factor_mean'),
                   '原始因子均值', 'factor_value')
        add_metric(distribution, 'std', self.stats.get('factor_std'),
                   '原始因子标准差', 'factor_value')
        add_metric(distribution, 'skew', self.stats.get('factor_skew'),
                   '原始因子偏度', 'ratio')
        add_metric(distribution, 'kurtosis', self.stats.get('factor_kurtosis'),
                   '原始因子超额峰度', 'ratio')

        rough_xml = ET.tostring(root, encoding='utf-8')
        xml_text = minidom.parseString(rough_xml).toprettyxml(
            indent='  ', encoding='utf-8').decode('utf-8')

        return xml_text

    def save_results(self, base_output_dir: str):
        """
        保存所有结果，包括性能摘要、时间序列数据和图表。
        参照 FactorComparator.save_results 的实现。
        """
        if self.stats is None:
            raise RuntimeError(
                "Please run the 'run()' method before saving results.")
        if not hasattr(self, 'figure') or self.figure is None:
            raise RuntimeError(
                "Please run the 'plot_results()' method before saving results."
            )

        # 直接使用 base_output_dir，不创建子目录
        output_dir = os.path.join(base_output_dir, str(self.name))
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to: {output_dir}")

        # 1. 保存绩效文本
        summary_path = os.path.join(output_dir, "performance_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(self._generate_stats_text())
        print(f"Performance summary saved to: {summary_path}")

        # 2. 保存时间序列数据为独立文件
        print("Saving time series data as separate files...")

        # 定义要保存的序列
        series_to_save = ['nav', 'ic', 'turnover']

        # 循环遍历每个指标，并单独保存
        for metric_name in series_to_save:
            if metric_name in self.resample_data.columns:
                # 构造文件名，例如: nav.csv
                file_name = f"{metric_name}.csv"
                file_path = os.path.join(output_dir, file_name)

                # 提取该序列并保存
                series = self.resample_data[metric_name]
                series.to_csv(file_path, header=True)
                print(f" -> Saved {file_path}")

        # 3. 保存图表
        image_path = os.path.join(output_dir, "evaluation_plot.png")
        self.figure.savefig(image_path, dpi=150)

        # 4. 保存xml
        xml_path = os.path.join(output_dir, "evaluation.xml")
        xml_text = self.generate_xml(
            self.resample_data['nav'].index[0].isoformat(),
            self.resample_data['nav'].index[-1].isoformat(),
        )
        with open(xml_path, 'w', encoding='utf-8') as file:
            file.write(xml_text)

        ##单独保存
        image_path1 = os.path.join(base_output_dir, "plot")
        os.makedirs(image_path1, exist_ok=True)
        self.figure.savefig(os.path.join(image_path1,
                                         "{0}.png".format(self.name)),
                            dpi=150)
        plt.close(self.figure)

        xml_path1 = os.path.join(base_output_dir, "xml")
        os.makedirs(xml_path1, exist_ok=True)
        xml_name1 = os.path.join(xml_path1, "{0}.xml".format(self.name))
        with open(xml_name1, 'w', encoding='utf-8') as file:
            file.write(xml_text)

        print(f"Evaluation plot saved to: {image_path}")
        print(f"Evaluation xml saved to: {xml_path}")
