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
        ic_str = f"{self.ic:.4f}" if self.category == BOTH_SIDE else "N/A"
        ir_str = f"{self.ir:.4f}" if self.category == BOTH_SIDE else "N/A"
        return (f"\n--- {self.category} ---"
                f"\nreturns_mean:{self.returns_mean:.6f}"
                f"\nreturns_std:{self.returns_std:.6f}"
                f"\nsharp:{self.sharp:.4f}"
                f"\nturnover:{self.turnover:.4f}"
                f"\nmaxdd:{self.maxdd:.4f}"
                f"\nreturns_mdd:{self.returns_mdd:.4f}"
                f"\nwin_rate:{self.win_rate:.4f}"
                f"\nic:{ic_str}"
                f"\nir:{ir_str}"
                f"\ncalmar:{self.calmar:.4f}"
                f"\nfitness:{self.fitness:.4f}"
                f"\ncount:{self.count:.1f}")


class MetricsTuple(
        namedtuple(
            'MetricsTuple',
            ('long_evaluate', 'short_evaluate', 'both_evaluate',
             'topn_evaluate', 'hold', 'freq', 'direction', 'bias', 'category',
             'top_n', 'quantile_evaluations', 'returns_type'))):
    __slots__ = ()

    def __repr__(self):
        return (f"long_evaluate:{self.long_evaluate}\n"
                f"short_evaluate:{self.short_evaluate}\n"
                f"both_evaluate:{self.both_evaluate}\n"
                f"topn_evaluate:{self.topn_evaluate}\n"
                f"hold:{self.hold}, freq:{self.freq}, "
                f"direction:{self.direction}, bias:{self.bias:.2f}, "
                f"category:{self.category}, top_n:{self.top_n}")

    def plot_results(self, title_prefix="Factor Evaluation", show=True):
        """
        绘制 ArbMetrics 评估器的四组策略 (Long, Short, Both, TopN) 对比报告。
        参照 cux001.py 的风格，但针对四线并发和大数据量(8760x100)进行极速优化的渲染。
        """
        # 确保必须是带序列运行 (is_series=True) 才能画图
        if self.long_evaluate.returns_series is None:
            raise ValueError("必须以 is_series=True 运行的 Metrics 才能进行画图！")

        import seaborn as sns

        # 解决 Matplotlib 中文乱码和负号显示问题
        plt.rcParams['font.sans-serif'] = [
            'Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans'
        ]
        plt.rcParams['axes.unicode_minus'] = False

        sns.set_style('whitegrid')

        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        fig.suptitle(
            f"{title_prefix} | TopN={self.top_n}, Hold={self.hold}, Freq={self.freq}",
            fontsize=18)

        def set_sequential_xticks(ax, series, num_ticks=7):
            """X轴时间刻度辅助方法"""
            tick_positions = np.linspace(0,
                                         len(series) - 1,
                                         num_ticks,
                                         dtype=int)
            if hasattr(series.index, 'strftime'):
                tick_labels = [
                    series.index[i].strftime('%Y-%m-%d')
                    for i in tick_positions
                ]
            else:
                tick_labels = [str(series.index[i]) for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=30, ha='right')

        # ------------------------------------------------------------------
        # 1. 净值曲线 (NAV) - 四组策略同台竞技
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]

        # 根据收益类型决定净值计算公式
        def calc_nav(series):
            s = series.fillna(0)
            if getattr(self, 'returns_type', 'log') == 'log':
                return np.exp(s.cumsum())
            else:
                return (1 + s).cumprod()

        nav_long = calc_nav(self.long_evaluate.returns_series)
        nav_short = calc_nav(self.short_evaluate.returns_series)
        nav_both = calc_nav(self.both_evaluate.returns_series)
        nav_topn = calc_nav(self.topn_evaluate.returns_series)

        ax1.plot(nav_long.values, label='Long NAV', color='red', alpha=0.8)
        ax1.plot(nav_short.values, label='Short NAV', color='blue', alpha=0.8)
        ax1.plot(nav_both.values,
                 label='Both NAV',
                 color='purple',
                 alpha=0.9,
                 linewidth=2)
        ax1.plot(nav_topn.values,
                 label='TopN NAV',
                 color='green',
                 alpha=0.9,
                 linewidth=2)

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

        def fmt_ic(val, is_both=False):
            if not is_both: return "N/A"
            if pd.isna(val): return "N/A"
            return f"{val:.4f}"

        stats_text = (
            f"{'Metric':<18} | {'Long':<14} | {'Short':<14} | {'Both':<14} | {'TopN':<14}\n"
            f"{'-'*78}\n"
            f"{'Ret Mean (Ann)':<18} | {fmt(self.long_evaluate.returns_mean,1):<14} | {fmt(self.short_evaluate.returns_mean,1):<14} | {fmt(self.both_evaluate.returns_mean,1):<14} | {fmt(self.topn_evaluate.returns_mean,1):<14}\n"
            f"{'Sharpe Ratio':<18} | {fmt(self.long_evaluate.sharp):<14} | {fmt(self.short_evaluate.sharp):<14} | {fmt(self.both_evaluate.sharp):<14} | {fmt(self.topn_evaluate.sharp):<14}\n"
            f"{'Max Drawdown':<18} | {fmt(self.long_evaluate.maxdd,1):<14} | {fmt(self.short_evaluate.maxdd,1):<14} | {fmt(self.both_evaluate.maxdd,1):<14} | {fmt(self.topn_evaluate.maxdd,1):<14}\n"
            f"{'Win Rate':<18} | {fmt(self.long_evaluate.win_rate,1):<14} | {fmt(self.short_evaluate.win_rate,1):<14} | {fmt(self.both_evaluate.win_rate,1):<14} | {fmt(self.topn_evaluate.win_rate,1):<14}\n"
            f"{'Turnover Mean':<18} | {fmt(self.long_evaluate.turnover):<14} | {fmt(self.short_evaluate.turnover):<14} | {fmt(self.both_evaluate.turnover):<14} | {fmt(self.topn_evaluate.turnover):<14}\n"
            f"{'Calmar':<18} | {fmt(self.long_evaluate.calmar):<14} | {fmt(self.short_evaluate.calmar):<14} | {fmt(self.both_evaluate.calmar):<14} | {fmt(self.topn_evaluate.calmar):<14}\n"
            f"{'IC Mean':<18} | {fmt_ic(self.long_evaluate.ic, False):<14} | {fmt_ic(self.short_evaluate.ic, False):<14} | {fmt_ic(self.both_evaluate.ic, True):<14} | {fmt_ic(self.topn_evaluate.ic, False):<14}\n"
            f"{'ICIR':<18} | {fmt_ic(self.long_evaluate.ir, False):<14} | {fmt_ic(self.short_evaluate.ir, False):<14} | {fmt_ic(self.both_evaluate.ir, True):<14} | {fmt_ic(self.topn_evaluate.ir, False):<14}\n"
            f"{'Fitness':<18} | {fmt(self.long_evaluate.fitness):<14} | {fmt(self.short_evaluate.fitness):<14} | {fmt(self.both_evaluate.fitness):<14} | {fmt(self.topn_evaluate.fitness):<14}\n"
            f"{'Avg Holding Count':<18} | {fmt(self.long_evaluate.count):<14} | {fmt(self.short_evaluate.count):<14} | {fmt(self.both_evaluate.count):<14} | {fmt(self.topn_evaluate.count):<14}\n"
        )
        ax_table.text(0.02,
                      0.95,
                      stats_text,
                      transform=ax_table.transAxes,
                      fontsize=12,
                      verticalalignment='top',
                      fontfamily='monospace',
                      bbox=dict(facecolor='white', alpha=0.8,
                                edgecolor='gray'))
        ax_table.set_title("Strategy Key Performance Indicators", fontsize=14)

        # ------------------------------------------------------------------
        # 3. 截面 IC 序列与累积图（只展示有意义的全市场 Both IC）
        # ------------------------------------------------------------------
        ax3 = axes[1, 0]
        ic_both = self.both_evaluate.ic_series.fillna(0)

        ax3.plot(ic_both.cumsum().values,
                 label='Factor CumIC (Both)',
                 color='purple',
                 alpha=0.9,
                 linewidth=2)

        set_sequential_xticks(ax3, ic_both)
        ax3.set_ylabel("Cumulative IC")
        ax3.set_title("Cross-sectional IC Analysis (Cumulative)")
        ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # ------------------------------------------------------------------
        # 4. 分位数累计收益曲线图 (Quantile Cumulative Returns)
        # ------------------------------------------------------------------
        ax4 = axes[1, 1]

        if getattr(self, 'quantile_evaluations', None):
            import matplotlib.cm as cm
            q_num = len(self.quantile_evaluations)
            colors = cm.coolwarm_r(np.linspace(0, 1, q_num))  # 红到蓝的渐变色谱

            # 画每条分位数的净值曲线 (NAV)
            for i, q_eval in enumerate(self.quantile_evaluations):
                if q_eval.returns_series is not None:
                    q_nav = calc_nav(q_eval.returns_series)
                    ax4.plot(q_nav.values,
                             label=q_eval.category,
                             color=colors[i],
                             alpha=0.85,
                             linewidth=1.5)

            if self.quantile_evaluations and self.quantile_evaluations[
                    0].returns_series is not None:
                set_sequential_xticks(
                    ax4, self.quantile_evaluations[0].returns_series)

            ax4.set_title(f"Quantile Cumulative Returns ({q_num} groups)")
            ax4.set_ylabel("NAV")

            # 把图例放在外侧避免遮挡严重区域
            ax4.legend(loc='lower left', prop={'size': 9})
            ax4.grid(True, alpha=0.5)
        else:
            ax4.text(0.5,
                     0.5,
                     "Quantile Evaluations Not Available",
                     ha='center',
                     va='center')
            ax4.axis('off')

        # ------------------------------------------------------------------
        # 5. 动态历史回撤对比区
        # ------------------------------------------------------------------
        ax5 = axes[2, 0]

        # 计算回撤
        def draw_dd(nav_series, name, c, ax):
            dd = (nav_series / nav_series.cummax() - 1) * 100
            ax.plot(dd.values, color=c, alpha=0.8, label=name)
            ax.fill_between(np.arange(len(dd)),
                            dd.values,
                            0,
                            color=c,
                            alpha=0.2)

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
        ax6.plot(turnover_long.rolling(smooth_win,
                                       min_periods=1).mean().values,
                 color='red',
                 alpha=0.6,
                 label='Long Turnover')
        ax6.plot(turnover_short.rolling(smooth_win,
                                        min_periods=1).mean().values,
                 color='blue',
                 alpha=0.6,
                 label='Short Turnover')
        ax6.plot(turnover_both.rolling(smooth_win,
                                       min_periods=1).mean().values,
                 color='purple',
                 alpha=0.7,
                 label='Both Turnover (Smoothed)')
        ax6.plot(turnover_topn.rolling(smooth_win,
                                       min_periods=1).mean().values,
                 color='green',
                 label='TopN Turnover (Smoothed)')

        set_sequential_xticks(ax6, turnover_topn)
        ax6.set_title("Turnover Over Time (Rolling Mean)")
        ax6.set_ylabel("Turnover")
        ax6.legend()
        ax6.grid(True)

        # 调整布局，防止标题/指标表格被裁切覆盖，同时抑制 UserWarning
        plt.subplots_adjust(top=0.92, bottom=0.08, wspace=0.2, hspace=0.35)

        if show:
            plt.show()

        return fig

    def to_dataframe(self):
        """
        提取非序列的核心指标，输出为 pandas DataFrame。
        行索引: ['long', 'short', 'both', 'topn']
        列名: ['returns_mean', 'returns_std', 'sharp', 'turnover', 'maxdd', 
               'returns_mdd', 'win_rate', 'ic', 'ir', 'calmar', 'fitness', 'count']
        """
        cols = [
            'returns_mean', 'returns_std', 'sharp', 'turnover', 'maxdd',
            'returns_mdd', 'win_rate', 'ic', 'ir', 'calmar', 'fitness', 'count'
        ]

        data = {}
        # 搜集主策略表现
        eval_list = [('long', self.long_evaluate),
                     ('short', self.short_evaluate),
                     ('both', self.both_evaluate),
                     ('topn', self.topn_evaluate)]

        # 追加分位数策略表现
        if getattr(self, 'quantile_evaluations', None):
            for q_eval in self.quantile_evaluations:
                eval_list.append((q_eval.category, q_eval))

        for row_name, evaluate in eval_list:
            row_data = {col: getattr(evaluate, col, np.nan) for col in cols}
            # 清理对于多空分组无统计学意义的噪音 IC 指标
            if row_name != 'both':
                row_data['ic'] = np.nan
                row_data['ir'] = np.nan
            data[row_name] = row_data

        return pd.DataFrame.from_dict(data, orient='index')

    def save_results(self,
                     base_output_dir: str,
                     title_prefix="Factor Evaluation",
                     image_export_dir: str = None):
        """
        保存所有结果，包括性能摘要、指标表格、时间序列数据和图表。
        如果指定了 image_export_dir，除了存到自身目录外，图表还会额外拷贝一份到该统一下，用来统一翻图。
        """
        import os
        os.makedirs(base_output_dir, exist_ok=True)
        print(f"Saving results to: {base_output_dir}")

        # 1. 保存绩效文本
        summary_path = os.path.join(base_output_dir, "performance_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"=== ArbMetrics Report [Cython] ===\n")
            f.write(
                f"Parameters: TopN={self.top_n}, Hold={self.hold}, Freq={self.freq}, Direction={self.direction}\n"
            )
            f.write(str(self.long_evaluate) + "\n")
            f.write(str(self.short_evaluate) + "\n")
            f.write(str(self.both_evaluate) + "\n")
            f.write(str(self.topn_evaluate) + "\n")
        print(f"Performance summary saved to: {summary_path}")

        # 2. 保存 DataFrame 版的核心指标为 CSV
        df_metrics = self.to_dataframe()
        metrics_csv_path = os.path.join(base_output_dir,
                                        "performance_metrics.csv")
        df_metrics.to_csv(metrics_csv_path,
                          header=True,
                          index_label='strategy')
        print(f"Metrics dataframe saved to: {metrics_csv_path}")

        # 3. 保存时间序列数据为独立文件
        if self.long_evaluate.returns_series is not None:
            print("Saving time series data as separate files...")
            eval_series_list = [('long', self.long_evaluate),
                                ('short', self.short_evaluate),
                                ('both', self.both_evaluate),
                                ('topn', self.topn_evaluate)]
            if getattr(self, 'quantile_evaluations', None):
                for q_eval in self.quantile_evaluations:
                    eval_series_list.append((q_eval.category, q_eval))

            for eval_name, evaluate in eval_series_list:
                df = pd.DataFrame()
                if evaluate.returns_series is not None:
                    df['returns'] = evaluate.returns_series
                    if getattr(self, 'returns_type', 'log') == 'log':
                        df['nav'] = np.exp(
                            evaluate.returns_series.fillna(0).cumsum())
                    else:
                        df['nav'] = (
                            1 + evaluate.returns_series.fillna(0)).cumprod()
                if getattr(evaluate, 'ic_series', None) is not None:
                    df['ic'] = evaluate.ic_series
                if getattr(evaluate, 'turnover_series', None) is not None:
                    df['turnover'] = evaluate.turnover_series
                if getattr(evaluate, 'count_series', None) is not None:
                    df['count'] = evaluate.count_series

                if not df.empty:
                    file_path = os.path.join(base_output_dir,
                                             f"{eval_name}_series.csv")
                    df.to_csv(file_path, header=True)
                    print(f" -> Saved {file_path}")

            # 3. 保存图表
            fig = self.plot_results(title_prefix=title_prefix, show=False)
            image_path = os.path.join(base_output_dir, "evaluation_plot.png")
            fig.savefig(image_path, dpi=300)

            # 如果配置了图库目录，单独再抽存一张图过去方便阅览
            if image_export_dir is not None:
                os.makedirs(image_export_dir, exist_ok=True)
                import re
                # 将可能带有非法字符的名字转为下划线，作为文件名
                safe_name = re.sub(r'[^\w\u4e00-\u9fa5\-]+', '_',
                                   title_prefix).strip('_')
                if not safe_name:
                    safe_name = "factor_plot"
                export_path = os.path.join(image_export_dir,
                                           f"{safe_name}.png")
                fig.savefig(export_path, dpi=300)
                print(f"Evaluation plot also exported to: {export_path}")

            plt.close(fig)
            print(f"Evaluation plot saved to: {image_path}")
        else:
            print(
                "Time series saving and plotting skipped because is_series=False."
            )


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
    def quick(cls,
              returns,
              factors,
              factor_name,
              hold=1,
              skip=0,
              dummy=None,
              category=EXCESS,
              show_log=False,
              is_series=False,
              save_file=None,
              max_points=2000,
              quantiles=0):
        """
        轻量评估：只计算 both 的 IC、ICIR 和 long 侧换手率。

        跳过 evaluate()、量化分组等全量计算，专为批量因子筛选设计，
        速度比 general() 快 3~5 倍。
        
        若 quantiles > 0, 也会一并算出分组净值并画在图上（需 is_series=True 且配置 save_dir）

        Returns
        -------
        dict: {'ic': float, 'icir': float, 'turnover': float}
        """
        booster = Booster(hold, skip, 20, category)

        # 预处理
        dummy_vals = dummy.values if dummy is not None else None
        ereturns = booster.yields(returns.values.copy(), dummy_vals, skip,
                                  category)
        
        score = booster.score(factors.values.copy(), dummy_vals)

        if ereturns is None or score is None:
            return {'ic': np.nan, 'icir': np.nan, 'turnover': np.nan}

        # 构建 both 权重
        right = booster.create_weight(score, is_pos=True)
        left = booster.create_weight(score, is_pos=False)

        # hold smoothing
        if hold > 1:
            idx = returns.index
            cols = returns.columns
            right = pd.DataFrame(right, index=idx, columns=cols)\
                      .rolling(hold, min_periods=1).sum().div(hold).values
            left  = pd.DataFrame(left,  index=idx, columns=cols)\
                      .rolling(hold, min_periods=1).sum().div(hold).values

        long_weight, _, both_weight, _ = booster.direction(
            right, left, ereturns)

        # IC / ICIR（only both）
        ic_arr, ic_mean, ic_std = booster.correlation(both_weight, ereturns,
                                                      'both')
        icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

        # 换手率（long 侧，反映单边真实换手）
        w0 = np.nan_to_num(long_weight, nan=0.0)
        turnover = float(
            np.mean(np.sum(np.abs(w0[1:] - w0[:-1]), axis=1) * 0.5))

        result = {
            'ic': round(ic_mean, 6),
            'icir': round(icir, 6),
            'turnover': round(turnover, 6)
        }

        # 量化分组收益（只算收益序列，不走全量evaluate），仅为画图准备
        quantile_series_list = []
        if is_series and quantiles > 0:
            pct_ranks = booster.percent_rank(score)
            for q in range(1, quantiles + 1):
                lower_bound = (q - 1) / quantiles
                upper_bound = q / quantiles
                mask = (pct_ranks > lower_bound) & (pct_ranks <= upper_bound)
                qw = np.where(mask, 1.0, 0.0)

                row_sums = np.sum(qw, axis=1, keepdims=True)
                with np.errstate(divide='ignore', invalid='ignore'):
                    qw = np.where(row_sums > 0, qw / row_sums, 0.0)

                if hold > 1:
                    qw = pd.DataFrame(qw).rolling(
                        hold, min_periods=1).sum().div(hold).values
                    row_sums_smooth = np.nansum(qw, axis=1, keepdims=True)
                    qw = np.where(row_sums_smooth > 0, qw / row_sums_smooth,
                                  0.0)

                # 计算分组收益序列
                q_rets = np.nansum(ereturns * qw, axis=1)
                quantile_series_list.append(
                    pd.Series(q_rets, index=returns.index))

        if show_log:
            print(
                f"[quick_ic] IC={ic_mean:.4f}  ICIR={icir:.4f}  turnover={turnover:.4f}"
            )

        # 如果需要时序并且有保存目录，绘制并发图
        if is_series and save_file is not None:
            import os
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.cm as cm

            # 解决 Matplotlib 中文乱码和负号显示问题
            plt.rcParams['font.sans-serif'] = [
                'Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans'
            ]
            plt.rcParams['axes.unicode_minus'] = False

            # 使用 returns 的 index
            ic_series = pd.Series(ic_arr, index=returns.index).fillna(0)
            cumsum_ic = ic_series.cumsum()

            # -------------------------------------------------------------
            # Downsample for faster plotting if data is too large (>2000 pts)
            # -------------------------------------------------------------
            total_points = len(ic_series)
            step = max(1, total_points // max_points)
            
            plot_idx = np.arange(0, total_points, step)
            if plot_idx[-1] != total_points - 1:
                plot_idx = np.append(plot_idx, total_points - 1)
                
            ic_series_plot = ic_series.iloc[plot_idx]
            cumsum_ic_plot = cumsum_ic.iloc[plot_idx]

            if quantiles > 0 and len(quantile_series_list) > 0:
                fig, (ax_q, ax1) = plt.subplots(2, 1, figsize=(12, 12))
                # 绘制上方的分组净值图
                colors = cm.coolwarm_r(np.linspace(0, 1, quantiles))
                for i, q_series in enumerate(quantile_series_list):
                    q_nav = np.exp(q_series.fillna(0).cumsum())
                    q_nav_plot = q_nav.iloc[plot_idx]
                    ax_q.plot(q_nav_plot.values,
                              label=f'Q{i+1}',
                              color=colors[i],
                              alpha=0.85,
                              linewidth=1.5)

                ax_q.set_title(
                    f"{factor_name} Cumulative Returns ({quantiles} groups)")
                ax_q.set_ylabel("NAV")
                ax_q.legend(loc='upper left', prop={'size': 9})
                ax_q.grid(True, alpha=0.5)

                # 设置 X 轴
                num_ticks = 7
                tick_positions = np.linspace(0,
                                             len(ic_series_plot) - 1,
                                             num_ticks,
                                             dtype=int)
                if hasattr(ic_series_plot.index, 'strftime'):
                    tick_labels = [
                        ic_series_plot.index[i].strftime('%Y-%m-%d')
                        for i in tick_positions
                    ]
                else:
                    tick_labels = [
                        str(ic_series_plot.index[i]) for i in tick_positions
                    ]
                ax_q.set_xticks(tick_positions)
                ax_q.set_xticklabels(tick_labels, rotation=30, ha='right')
            else:
                fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.bar(np.arange(len(ic_series_plot)),
                    ic_series_plot.values,
                    label='Rolling IC (Both)',
                    color='steelblue',
                    alpha=0.7,
                    width=1.0)
            ax1.set_ylabel("Rolling Spearman IC", color='steelblue')
            ax1.axhline(0, color='gray', linestyle='--', linewidth=1)

            ax2 = ax1.twinx()
            ax2.plot(cumsum_ic_plot.values,
                     label='Cumulative IC',
                     color='purple',
                     linewidth=2)
            ax2.set_ylabel("Cumulative IC", color='purple')

            # X 轴时间刻度辅助
            if 'num_ticks' not in locals():
                num_ticks = 7
                tick_positions = np.linspace(0,
                                             len(ic_series_plot) - 1,
                                             num_ticks,
                                             dtype=int)
                if hasattr(ic_series_plot.index, 'strftime'):
                    tick_labels = [
                        ic_series_plot.index[i].strftime('%Y-%m-%d')
                        for i in tick_positions
                    ]
                else:
                    tick_labels = [
                        str(ic_series_plot.index[i]) for i in tick_positions
                    ]
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels(tick_labels, rotation=30, ha='right')

            # 合并图例
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2,
                       labels_1 + labels_2,
                       loc='upper left')

            ax1.set_title("Quick Factor IC Analysis")
            fig.tight_layout()

            #save_file = os.path.join(save_dir, "quick_ic_plot.png")
            fig.savefig(save_file, dpi=300)
            plt.close(fig)
            if show_log:
                print(f"Quick IC & Quantile plot saved to: {save_file}")

        return result

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
                topn_weight_method='factor',
                quantiles=5,
                returns_type='log',
                annual_days=None):
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
                      topn_weight_method=topn_weight_method,
                      quantiles=quantiles,
                      returns_type=returns_type,
                      annual_days=annual_days)
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
                 topn_weight_method='factor',
                 quantiles=5,
                 returns_type='log',
                 annual_days=None):
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
        self.quantiles = quantiles
        self.returns_type = returns_type
        self.annual_days = annual_days

        # 保存 pandas 引用 (用于输出)
        self._returns_index = returns.index
        self._returns_columns = returns.columns

        # 生成适用于 Cython 的交易日归集组别 (将高频 timestamp 映射到不同天的 Int Index)
        if self.annual_days is not None:
            if hasattr(self._returns_index, 'date'):
                dates = self._returns_index.date
            else:
                dates = pd.to_datetime(self._returns_index).date
            groups, uniques = pd.factorize(dates)
            self._date_groups = groups.astype(np.int32)
            self._num_groups = len(uniques)
        else:
            self._date_groups = None
            self._num_groups = 0

        # 创建 Cython Booster
        self.booster = Booster(hold, skip, top_n, category)

        # 转 numpy 并预处理
        dummy_vals = dummy.values if dummy is not None else None
        self.ereturns = self.booster.yields(returns.values.copy(), dummy_vals,
                                            skip, category)
        self.score_vals = self.booster.score(factors.values.copy(), dummy_vals)

        if self.ereturns is not None and self.score_vals is not None:
            self.valid = True

    def _make_evaluate_tuple(self, indicator, ic_arr, ic_mean, ic_std, weight,
                             category):
        """将 booster 输出转换为 EvaluateTuple"""
        (rets_sum, rets_mean, rets_std, sharp, turnover, maxdd, ret2mdd,
         calmar, win_rate, fitness, turnover_series, count_series) = indicator

        ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
        count = np.mean(count_series)

        # 可选时序输出
        if self.is_series:
            returns_series = pd.Series(rets_sum,
                                       index=self._returns_index,
                                       name='returns')
            ic_series = pd.Series(ic_arr, index=self._returns_index, name='ic')

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

        return EvaluateTuple(returns_mean=rets_mean,
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

    def _apply_fee(self, returns, weight):
        """
        扣除交易费用的等效广播技术。
        为了不修改底层的 Cython 代码结构并保证极速运行，将每个 period 组合层面产生的
        总交易成本（fee * turnover_t），根据对应的绝对权重均匀分摊成各个 active asset 的价格下修惩罚。
        """
        if self.fee <= 0:
            return returns

        weight0 = np.nan_to_num(weight, nan=0.0)
        # 计算每个 period 的 turnover
        tv = np.nansum(np.abs(weight0[1:] - weight0[:-1]), axis=1) * 0.5
        tv_full = np.concatenate([[0.0], tv])  # Shape: (t,)

        # 计算每个 period 总绝对持仓权重 sum(|w|)
        W_abs = np.nansum(np.abs(weight0), axis=1)  # Shape: (t,)

        # 计算单位绝对权重需要分摊的手续费系数
        fee_ratio = np.zeros_like(W_abs)
        mask = W_abs > 1e-10
        fee_ratio[mask] = (self.fee * tv_full[mask]) / W_abs[mask]

        # 构建惩罚：多头收益率降低 (r - fee)，空头收益上升导致损失增大 (r + fee)
        # 巧妙利用 np.sign(w) 保证方向性惩罚正确
        w_sign = np.sign(weight0)
        iret = returns - (w_sign * fee_ratio[:, np.newaxis])

        return iret

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
        # 计算扣除手续费后的收益率序列 (重要修正)
        eval_freq = self.annual_days if self.annual_days is not None else self.freq

        iret_long = self._apply_fee(self.ereturns, long_weight)
        long_ind = self.booster.evaluate(long_weight, iret_long, self.hold,
                                         eval_freq, self._date_groups,
                                         self._num_groups)
        long_ic, long_ic_mean, long_ic_std = self.booster.correlation(
            long_weight, self.ereturns, 'long')
        long_evaluate = self._make_evaluate_tuple(long_ind, long_ic,
                                                  long_ic_mean, long_ic_std,
                                                  long_weight, OLNY_LONG)

        iret_short = self._apply_fee(self.ereturns, short_weight)
        short_ind = self.booster.evaluate(short_weight, iret_short, self.hold,
                                          eval_freq, self._date_groups,
                                          self._num_groups)
        short_ic, short_ic_mean, short_ic_std = self.booster.correlation(
            short_weight, self.ereturns, 'short')
        short_evaluate = self._make_evaluate_tuple(short_ind, short_ic,
                                                   short_ic_mean, short_ic_std,
                                                   short_weight, OLNY_SHORT)

        iret_both = self._apply_fee(self.ereturns, both_weight)
        both_ind = self.booster.evaluate(both_weight, iret_both, self.hold,
                                         eval_freq, self._date_groups,
                                         self._num_groups)
        both_ic, both_ic_mean, both_ic_std = self.booster.correlation(
            both_weight, self.ereturns, 'both')
        both_evaluate = self._make_evaluate_tuple(both_ind, both_ic,
                                                  both_ic_mean, both_ic_std,
                                                  both_weight, BOTH_SIDE)

        # ---------- TopN ----------
        topn_weight = self.booster.create_topn_weight(score, self.top_n,
                                                      self.topn_weight_method)
        topn_weight = self._apply_hold_smoothing(topn_weight)

        # re-normalize after smoothing
        if self.hold > 1:
            sums = np.nansum(topn_weight, axis=1, keepdims=True)
            topn_weight = np.divide(topn_weight,
                                    sums,
                                    where=sums > 0,
                                    out=topn_weight)

        iret_topn = self._apply_fee(self.ereturns, topn_weight)
        topn_ind = self.booster.evaluate(topn_weight, iret_topn, self.hold,
                                         eval_freq, self._date_groups,
                                         self._num_groups)
        topn_ic, topn_ic_mean, topn_ic_std = self.booster.correlation(
            topn_weight, self.ereturns, 'long')
        topn_evaluate = self._make_evaluate_tuple(topn_ind, topn_ic,
                                                  topn_ic_mean, topn_ic_std,
                                                  topn_weight, TOP_N)

        # ---------- Quantiles ----------
        quantile_evals = []
        if self.quantiles > 0:
            # Cython accelerated percent rank (replaces pd.DataFrame.rank)
            pct_ranks = self.booster.percent_rank(score)
            for q in range(1, self.quantiles + 1):
                lower_bound = (q - 1) / self.quantiles
                upper_bound = q / self.quantiles

                # Filter indices within percentile borders
                mask = (pct_ranks > lower_bound) & (pct_ranks <= upper_bound)
                qw = np.where(mask, 1.0, 0.0)

                # Avoid division by zero
                row_sums = np.sum(qw, axis=1, keepdims=True)
                qw = np.divide(qw, row_sums, where=row_sums > 0, out=qw)

                qw = self._apply_hold_smoothing(qw)
                # Re-normalize after smoothing (equal weight within quantile group)
                if self.hold > 1:
                    row_sums_smooth = np.nansum(qw, axis=1, keepdims=True)
                    qw = np.divide(qw,
                                   row_sums_smooth,
                                   where=row_sums_smooth > 0,
                                   out=qw)

                iret_qw = self._apply_fee(self.ereturns, qw)
                q_ind = self.booster.evaluate(qw, iret_qw, self.hold,
                                              eval_freq, self._date_groups,
                                              self._num_groups)
                # IC isn't strictly necessary for local quantiles, but we calculate it for tuple completeness
                q_ic, q_ic_mean, q_ic_std = self.booster.correlation(
                    qw, self.ereturns, 'long')
                q_eval = self._make_evaluate_tuple(q_ind, q_ic, q_ic_mean,
                                                   q_ic_std, qw, f'Q{q}')
                quantile_evals.append(q_eval)

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
                  short_evaluate.count if short_evaluate.count != 0 else 0),
            category=self.category,
            direction=self.direction,
            top_n=self.top_n,
            quantile_evaluations=quantile_evals,
            returns_type=self.returns_type)
