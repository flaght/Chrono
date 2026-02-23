# -*- encoding:utf-8 -*-
"""
Cython 加速版 ArbMetrics 评估器

架构：
    metrics.py (Python 层) — 负责 pandas ↔ numpy 转换 + 高层逻辑
    booster.pyx (Cython 层) — 负责核心数值计算
"""
import pdb
import functools
import numpy as np
import pandas as pd
from collections import namedtuple
from .booster import Booster
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DALIY_PER_YEAR = 252
WEEKLY_PER_YEAR = 52
MONTHLY_PER_YEAR = 12
QUARTERLY_PER_YEAR = 4
YEARLY_PER_YEAR = 1
HOURLY_PER_YEAR = 365 * 24  # 8760

OLNY_LONG = 'long'
OLNY_SHORT = 'short'
BOTH_SIDE = 'both'
TOP_N = 'topn'

POSITIVE = 1
NEGATIVE = -1

EXCESS = 1
ABSOLUTE = -1


class EvaluateTuple(
        namedtuple('EvaluateTuple',
                   ('returns_mean', 'returns_std', 'sharp', 'turnover',
                    'maxdd', 'returns_mdd', 'win_rate', 'ic', 'ir', 'fitness',
                    'category', 'count', 'calmar', 'count_series',
                    'returns_series', 'ic_series', 'turnover_series'))):

    __slots__ = ()

    def __repr__(self):
        return (f"\n--- {self.category} ---"
                f"\nreturns_mean:{self.returns_mean:.6f}"
                f"\nreturns_std:{self.returns_std:.6f}"
                f"\nsharp:{self.sharp:.4f}"
                f"\nturnover:{self.turnover:.4f}"
                f"\nmaxdd:{self.maxdd:.4f}"
                f"\nreturns_mdd:{self.returns_mdd:.4f}"
                f"\nwin_rate:{self.win_rate:.4f}"
                f"\nic:{self.ic:.4f}"
                f"\nir:{self.ir:.4f}"
                f"\calmar:{self.calmar:.4f}"
                f"\nfitness:{self.fitness:.4f}"
                f"\ncount:{self.count:.1f}")


class MetricsTuple(
        namedtuple('MetricsTuple',
                   ('long_evaluate', 'short_evaluate', 'both_evaluate',
                    'topn_evaluate', 'hold', 'freq', 'direction', 'bias',
                    'category', 'top_n'))):
    __slots__ = ()

    def __repr__(self):
        return (f"long_evaluate:{self.long_evaluate}\n"
                f"short_evaluate:{self.short_evaluate}\n"
                f"both_evaluate:{self.both_evaluate}\n"
                f"topn_evaluate:{self.topn_evaluate}\n"
                f"hold:{self.hold}, freq:{self.freq}, "
                f"direction:{self.direction}, bias:{self.bias:.2f}, "
                f"category:{self.category}, top_n:{self.top_n}")

    def plot_results(self, title_prefix="Factor Evaluation"):
        """
        绘制 ArbMetrics 评估器的四组策略 (Long, Short, Both, TopN) 对比报告。
        参照 cux001.py 的风格，但针对四线并发和大数据量(8760x100)进行极速优化的渲染。
        """
        # 确保必须是带序列运行 (is_series=True) 才能画图
        if self.long_evaluate.returns_series is None:
            raise ValueError("必须以 is_series=True 运行的 Metrics 才能进行画图！")

        import seaborn as sns
        
        # 解决 Matplotlib 中文乱码和负号显示问题
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        sns.set_style('whitegrid')

        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        fig.suptitle(f"{title_prefix} | TopN={self.top_n}, Hold={self.hold}, Freq={self.freq}", fontsize=18)

        def set_sequential_xticks(ax, series, num_ticks=7):
            """X轴时间刻度辅助方法"""
            tick_positions = np.linspace(0, len(series) - 1, num_ticks, dtype=int)
            if hasattr(series.index, 'strftime'):
                tick_labels = [series.index[i].strftime('%Y-%m-%d') for i in tick_positions]
            else:
                tick_labels = [str(series.index[i]) for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=30, ha='right')

        # ------------------------------------------------------------------
        # 1. 净值曲线 (NAV) - 四组策略同台竞技
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        # 计算 cumulative return NAV (1+r).cumprod()
        nav_long = (1 + self.long_evaluate.returns_series.fillna(0)).cumprod()
        nav_short = (1 + self.short_evaluate.returns_series.fillna(0)).cumprod()
        nav_both = (1 + self.both_evaluate.returns_series.fillna(0)).cumprod()
        nav_topn = (1 + self.topn_evaluate.returns_series.fillna(0)).cumprod()

        ax1.plot(nav_long.values, label='Long NAV', color='red', alpha=0.8)
        ax1.plot(nav_short.values, label='Short NAV', color='blue', alpha=0.8)
        ax1.plot(nav_both.values, label='Both (L-S) NAV', color='purple', alpha=0.9, linewidth=2)
        ax1.plot(nav_topn.values, label='TopN NAV', color='green', alpha=0.9, linewidth=2)

        set_sequential_xticks(ax1, nav_long)
        ax1.set_title("Performance (NAV)")
        ax1.set_ylabel("NAV")
        ax1.legend()
        ax1.grid(True)

        # ------------------------------------------------------------------
        # 2. 绩效指标表格对比 (KPIs)
        # ------------------------------------------------------------------
        ax_table = axes[0, 1]
        ax_table.axis('off')
        
        # 提取指标格式化
        def fmt(val, is_pct=False):
            if pd.isna(val): return "N/A"
            return f"{val:.2%}" if is_pct else f"{val:.4f}"

        stats_text = (
            f"{'Metric':<18} | {'Long':<14} | {'Short':<14} | {'Both':<14} | {'TopN':<14}\n"
            f"{'-'*78}\n"
            f"{'Ret Mean (Ann)':<18} | {fmt(self.long_evaluate.returns_mean,1):<14} | {fmt(self.short_evaluate.returns_mean,1):<14} | {fmt(self.both_evaluate.returns_mean,1):<14} | {fmt(self.topn_evaluate.returns_mean,1):<14}\n"
            f"{'Sharpe Ratio':<18} | {fmt(self.long_evaluate.sharp):<14} | {fmt(self.short_evaluate.sharp):<14} | {fmt(self.both_evaluate.sharp):<14} | {fmt(self.topn_evaluate.sharp):<14}\n"
            f"{'Max Drawdown':<18} | {fmt(self.long_evaluate.maxdd,1):<14} | {fmt(self.short_evaluate.maxdd,1):<14} | {fmt(self.both_evaluate.maxdd,1):<14} | {fmt(self.topn_evaluate.maxdd,1):<14}\n"
            f"{'Win Rate':<18} | {fmt(self.long_evaluate.win_rate,1):<14} | {fmt(self.short_evaluate.win_rate,1):<14} | {fmt(self.both_evaluate.win_rate,1):<14} | {fmt(self.topn_evaluate.win_rate,1):<14}\n"
            f"{'Turnover Mean':<18} | {fmt(self.long_evaluate.turnover):<14} | {fmt(self.short_evaluate.turnover):<14} | {fmt(self.both_evaluate.turnover):<14} | {fmt(self.topn_evaluate.turnover):<14}\n"
            f"{'Calmar':<18} | {fmt(self.long_evaluate.calmar):<14} | {fmt(self.short_evaluate.calmar):<14} | {fmt(self.both_evaluate.calmar):<14} | {fmt(self.topn_evaluate.calmar):<14}\n"
            f"{'IC Mean':<18} | {fmt(self.long_evaluate.ic):<14} | {fmt(self.short_evaluate.ic):<14} | {fmt(self.both_evaluate.ic):<14} | {fmt(self.topn_evaluate.ic):<14}\n"
            f"{'ICIR':<18} | {fmt(self.long_evaluate.ir):<14} | {fmt(self.short_evaluate.ir):<14} | {fmt(self.both_evaluate.ir):<14} | {fmt(self.topn_evaluate.ir):<14}\n"
            f"{'Fitness':<18} | {fmt(self.long_evaluate.fitness):<14} | {fmt(self.short_evaluate.fitness):<14} | {fmt(self.both_evaluate.fitness):<14} | {fmt(self.topn_evaluate.fitness):<14}\n"
            f"{'Avg Holding Count':<18} | {fmt(self.long_evaluate.count):<14} | {fmt(self.short_evaluate.count):<14} | {fmt(self.both_evaluate.count):<14} | {fmt(self.topn_evaluate.count):<14}\n"
        )
        ax_table.text(0.02, 0.95, stats_text, transform=ax_table.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        ax_table.set_title("Strategy Key Performance Indicators", fontsize=14)

        # ------------------------------------------------------------------
        # 3. 截面 IC 序列与累积图（取多头/TopN 代表性特征）
        # ------------------------------------------------------------------
        ax3 = axes[1, 0]
        # 展示4条累计IC曲线
        ic_long = self.long_evaluate.ic_series.fillna(0)
        ic_short = self.short_evaluate.ic_series.fillna(0)
        ic_both = self.both_evaluate.ic_series.fillna(0)
        ic_topn = self.topn_evaluate.ic_series.fillna(0)

        ax3.plot(ic_long.cumsum().values, label='Long CumIC', color='red', alpha=0.8)
        ax3.plot(ic_short.cumsum().values, label='Short CumIC', color='blue', alpha=0.8)
        ax3.plot(ic_both.cumsum().values, label='Both CumIC', color='purple', alpha=0.9, linewidth=2)
        ax3.plot(ic_topn.cumsum().values, label='TopN CumIC', color='green', alpha=0.9, linewidth=2)

        set_sequential_xticks(ax3, ic_long)
        ax3.set_ylabel("Cumulative IC")
        ax3.set_title("Cross-sectional IC Analysis (Cumulative)")
        ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # ------------------------------------------------------------------
        # 4. 每日收益率分布取代臃肿散点图 (提高渲染效率)
        # ------------------------------------------------------------------
        ax4 = axes[1, 1]
        ax4.hist(self.topn_evaluate.returns_series.dropna().values, bins=60, alpha=0.6, color='green', label='TopN Returns')
        ax4.hist(self.both_evaluate.returns_series.dropna().values, bins=60, alpha=0.5, color='purple', label='Both Returns')
        ax4.set_title("Daily Portfolio Returns Distribution")
        ax4.set_xlabel("Return")
        ax4.set_ylabel("Frequency")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # ------------------------------------------------------------------
        # 5. 动态历史回撤对比区
        # ------------------------------------------------------------------
        ax5 = axes[2, 0]
        # 计算回撤
        def draw_dd(nav_series, name, c, ax):
            dd = (nav_series / nav_series.cummax() - 1) * 100
            ax.plot(dd.values, color=c, alpha=0.8, label=name)
            ax.fill_between(np.arange(len(dd)), dd.values, 0, color=c, alpha=0.2)
            
        draw_dd(nav_both, "Both Drawdown", "purple", ax5)
        draw_dd(nav_topn, "TopN Drawdown", "green", ax5)

        set_sequential_xticks(ax5, nav_topn)
        ax5.set_title(f"Drawdown Over Time")
        ax5.set_ylabel("Drawdown (%)")
        ax5.set_ylim(bottom=None, top=0.5)
        ax5.legend()
        ax5.grid(True)

        # ------------------------------------------------------------------
        # 6. 换手率监控
        # ------------------------------------------------------------------
        ax6 = axes[2, 1]
        # 去掉初始位置 (第一天)，避免初始建仓时的换手率峰值影响图表比例
        turnover_long = self.long_evaluate.turnover_series.fillna(0).iloc[1:]
        turnover_short = self.short_evaluate.turnover_series.fillna(0).iloc[1:]
        turnover_both = self.both_evaluate.turnover_series.fillna(0).iloc[1:]
        turnover_topn = self.topn_evaluate.turnover_series.fillna(0).iloc[1:]

        # 平滑换手率展示，避免刺破天际看不清趋势
        smooth_win = max(1, len(turnover_topn) // 100)
        ax6.plot(turnover_long.rolling(smooth_win, min_periods=1).mean().values, color='red', alpha=0.6, label='Long Turnover')
        ax6.plot(turnover_short.rolling(smooth_win, min_periods=1).mean().values, color='blue', alpha=0.6, label='Short Turnover')
        ax6.plot(turnover_both.rolling(smooth_win, min_periods=1).mean().values, color='purple', alpha=0.7, label='Both Turnover (Smoothed)')
        ax6.plot(turnover_topn.rolling(smooth_win, min_periods=1).mean().values, color='green', label='TopN Turnover (Smoothed)')
        
        set_sequential_xticks(ax6, turnover_topn)
        ax6.set_title("Turnover Over Time (Rolling Mean)")
        ax6.set_ylabel("Turnover")
        ax6.legend()
        ax6.grid(True)

        # 调整布局，防止标题/指标表格被裁切覆盖，同时抑制 UserWarning
        plt.subplots_adjust(top=0.92, bottom=0.08, wspace=0.2, hspace=0.35)
        
        plt.show()


def valid_check(func):
    """检测度量的输入是否正常"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.valid:
            return func(self, *args, **kwargs)
        else:
            print('[WARN] Metrics input is invalid or zero order gen!')
            return None

    return wrapper


class Metrics(object):
    """
    Cython 加速版因子截面评估器。
    
    用法与 csm001.Metrics 完全相同，核心计算由 Cython Booster 执行。
    """

    @classmethod
    def general(cls,
                returns,
                factors,
                hold=1,
                skip=0,
                top_n=20,
                dummy=None,
                direction=None,
                category=EXCESS,
                freq=DALIY_PER_YEAR,
                fee=0.0,
                show_log=True,
                is_series=False,
                topn_weight_method='factor'):
        """工厂方法：一行调用完成全量评估。"""
        metrics = cls(returns=returns,
                      factors=factors,
                      hold=hold,
                      freq=freq,
                      direction=direction,
                      category=category,
                      skip=skip,
                      top_n=top_n,
                      dummy=dummy,
                      fee=fee,
                      show_log=show_log,
                      is_series=is_series,
                      topn_weight_method=topn_weight_method)
        return metrics.fit_metrics()

    def __init__(self,
                 returns,
                 factors,
                 hold=1,
                 direction=None,
                 category=EXCESS,
                 freq=DALIY_PER_YEAR,
                 dummy=None,
                 skip=0,
                 top_n=20,
                 fee=0.0,
                 show_log=True,
                 is_series=False,
                 topn_weight_method='factor'):
        self.valid = False
        self.category = category
        self.skip = skip
        self.top_n = top_n
        self.fee = fee
        self.freq = freq
        self.hold = hold
        self.show_log = show_log
        self.is_series = is_series
        self.direction = direction
        self.topn_weight_method = topn_weight_method

        # 保存 pandas 引用 (用于输出)
        self._returns_index = returns.index
        self._returns_columns = returns.columns

        # 创建 Cython Booster
        self.booster = Booster(hold, skip, top_n, category)

        # 转 numpy 并预处理
        dummy_vals = dummy.values if dummy is not None else None
        self.ereturns = self.booster.yields(returns.values.copy(), dummy_vals,
                                            skip, category)
        self.score_vals = self.booster.score(
            factors.values.copy(),
            dummy_vals)

        if self.ereturns is not None and self.score_vals is not None:
            self.valid = True

    def _make_evaluate_tuple(self, indicator, ic_arr, ic_mean, ic_std,
                             weight, category):
        """将 booster 输出转换为 EvaluateTuple"""
        (rets_sum, rets_mean, rets_std, sharp, turnover, maxdd,
         ret2mdd, calmar, win_rate, fitness, turnover_series,
         count_series) = indicator

        ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
        count = np.mean(count_series)

        # 可选时序输出
        if self.is_series:
            returns_series = pd.Series(rets_sum,
                                       index=self._returns_index,
                                       name='returns')
            ic_series = pd.Series(ic_arr,
                                  index=self._returns_index,
                                  name='ic')
            
            # turnover has Length T-1, so we pad the first value as 0 to match T
            padded_tv = np.concatenate([[0.0], turnover_series])
            tv_series = pd.Series(padded_tv,
                                  index=self._returns_index,
                                  name='turnover')
            cnt_series = pd.Series(count_series,
                                   index=self._returns_index,
                                   name='count')
        else:
            returns_series = None
            ic_series = None
            tv_series = None
            cnt_series = None

        return EvaluateTuple(
            returns_mean=rets_mean,
            returns_std=rets_std,
            sharp=sharp,
            turnover=turnover,
            maxdd=maxdd,
            returns_mdd=ret2mdd,
            win_rate=win_rate,
            ic=ic_mean,
            ir=ir,
            calmar=calmar,
            fitness=fitness,
            count=count,
            category=category,
            count_series=cnt_series,
            returns_series=returns_series,
            ic_series=ic_series,
            turnover_series=tv_series)

    def _apply_hold_smoothing(self, weight):
        """apply rolling hold smoothing on numpy weight"""
        if self.hold > 1:
            weight_df = pd.DataFrame(weight,
                                     index=self._returns_index,
                                     columns=self._returns_columns)
            weight_df = weight_df.rolling(self.hold,
                                          min_periods=1).sum() / self.hold
            return weight_df.values
        return weight

    def _apply_fee(self, rets_sum, weight):
        """扣除交易费用"""
        if self.fee > 0:
            tv = np.nansum(np.abs(weight[1:] - weight[:-1]), axis=1) * 0.5
            tv_full = np.concatenate([[0.0], tv])
            rets_sum = rets_sum - self.fee * tv_full
        return rets_sum

    @valid_check
    def fit_metrics(self):
        """执行全量评估"""
        score = self.score_vals

        # ---------- Long / Short / Both ----------
        right = self.booster.create_weight(score, is_pos=True)
        left = self.booster.create_weight(score, is_pos=False)

        right = self._apply_hold_smoothing(right)
        left = self._apply_hold_smoothing(left)

        long_weight, short_weight, both_weight, direction_val = \
            self.booster.direction(right, left, self.ereturns)

        if self.direction is not None:
            direction_val = self.direction

        # Evaluate each side
        long_ind = self.booster.evaluate(long_weight, self.ereturns,
                                         self.hold, self.freq)
        long_ic, long_ic_mean, long_ic_std = self.booster.correlation(
            long_weight, self.ereturns, 'long')
        long_evaluate = self._make_evaluate_tuple(
            long_ind, long_ic, long_ic_mean, long_ic_std,
            long_weight, OLNY_LONG)

        short_ind = self.booster.evaluate(short_weight, self.ereturns,
                                          self.hold, self.freq)
        short_ic, short_ic_mean, short_ic_std = self.booster.correlation(
            short_weight, self.ereturns, 'short')
        short_evaluate = self._make_evaluate_tuple(
            short_ind, short_ic, short_ic_mean, short_ic_std,
            short_weight, OLNY_SHORT)

        both_ind = self.booster.evaluate(both_weight, self.ereturns,
                                         self.hold, self.freq)
        both_ic, both_ic_mean, both_ic_std = self.booster.correlation(
            both_weight, self.ereturns, 'both')
        both_evaluate = self._make_evaluate_tuple(
            both_ind, both_ic, both_ic_mean, both_ic_std,
            both_weight, BOTH_SIDE)

        # ---------- TopN ----------
        topn_weight = self.booster.create_topn_weight(
            score, self.top_n, self.topn_weight_method)
        topn_weight = self._apply_hold_smoothing(topn_weight)

        # re-normalize after smoothing
        if self.hold > 1:
            sums = np.nansum(topn_weight, axis=1, keepdims=True)
            topn_weight = np.divide(topn_weight, sums,
                                    where=sums > 0, out=topn_weight)

        topn_ind = self.booster.evaluate(topn_weight, self.ereturns,
                                         self.hold, self.freq)
        topn_ic, topn_ic_mean, topn_ic_std = self.booster.correlation(
            topn_weight, self.ereturns, 'long')
        topn_evaluate = self._make_evaluate_tuple(
            topn_ind, topn_ic, topn_ic_mean, topn_ic_std,
            topn_weight, TOP_N)

        self.direction = direction_val

        if self.show_log:
            print(f"\n{'=' * 50}")
            print(f"  ArbMetrics Report [Cython]  (top_n={self.top_n}, "
                  f"hold={self.hold}, freq={self.freq})")
            print(f"{'=' * 50}")
            print(long_evaluate)
            print(short_evaluate)
            print(both_evaluate)
            print(topn_evaluate)
            print(f"\ndirection={self.direction}")
            print(f"{'=' * 50}")

        return MetricsTuple(
            long_evaluate=long_evaluate,
            short_evaluate=short_evaluate,
            both_evaluate=both_evaluate,
            topn_evaluate=topn_evaluate,
            freq=self.freq,
            hold=self.hold,
            bias=(long_evaluate.count /
                  short_evaluate.count
                  if short_evaluate.count != 0 else 0),
            category=self.category,
            direction=self.direction,
            top_n=self.top_n)
