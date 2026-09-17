import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.cux001 import FactorEvaluate1 as FactorEvaluate1001


class FactorEvaluate1(FactorEvaluate1001):
    """在 cux001 基础上补充信号专用统计。

    与因子评估器的关键差异：
    1. 一次只允许评价一个品种，避免换手、IC和收益跨品种串联。
    2. 禁止根据评价区间 IC 自动反转信号。
    3. ``win_rate`` 表示有仓位时点胜率，并额外返回信号覆盖率、
       多空胜率、仓位分布和交易成本。
    """

    def __init__(self,
                 factor_data: pd.DataFrame,
                 resampling_win: int = 1,
                 factor_name: str = "signal",
                 code: str = "",
                 ret_name: str = "ret",
                 roll_win: int = 252,
                 fee: float = 0.0003,
                 scale_method: str = "raw",
                 annualization_factor: int = 252,
                 expression=None,
                 name=None):
        self.code = code
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

    @staticmethod
    def _safe_mean(mask, values):
        if not mask.any():
            return np.nan
        return float(values.loc[mask].mean())

    def cal_pnl(self, position_epsilon: float = 1e-12):
        """复用父类盈亏逻辑，并补充信号专用评价指标。"""
        stats = super(FactorEvaluate1, self).cal_pnl()

        position = self.resample_data["pos"]
        net_ret = self.resample_data["net_ret"]
        gross_ret = self.resample_data["gross_ret"]
        turnover = self.resample_data["turnover"]

        active = position.abs() > position_epsilon
        long_mask = position > position_epsilon
        short_mask = position < position_epsilon
        flat_mask = ~active

        # 父类 win_rate 把空仓时点也放入分母。信号评估时将 win_rate
        # 定义为有仓位时点胜率，同时保留原口径为 time_win_rate。
        stats["time_win_rate"] = stats["win_rate"]
        stats["win_rate"] = self._safe_mean(active, net_ret > 0)
        stats["active_win_rate"] = stats["win_rate"]
        stats["signal_rate"] = float(active.mean())
        stats["flat_rate"] = float(flat_mask.mean())
        stats["long_rate"] = float(long_mask.mean())
        stats["short_rate"] = float(short_mask.mean())
        stats["avg_abs_position"] = float(position.abs().mean())

        stats["long_count"] = int(long_mask.sum())
        stats["short_count"] = int(short_mask.sum())
        stats["flat_count"] = int(flat_mask.sum())
        stats["long_win_rate"] = self._safe_mean(long_mask, net_ret > 0)
        stats["short_win_rate"] = self._safe_mean(short_mask, net_ret > 0)
        stats["long_avg_ret"] = self._safe_mean(long_mask, net_ret)
        stats["short_avg_ret"] = self._safe_mean(short_mask, net_ret)
        stats["long_total_ret"] = float(net_ret.loc[long_mask].sum())
        stats["short_total_ret"] = float(net_ret.loc[short_mask].sum())

        cost = self.fee * turnover
        stats["gross_ret_sum"] = float(gross_ret.sum())
        stats["net_ret_sum"] = float(net_ret.sum())
        stats["total_cost"] = float(cost.sum())
        stats["avg_cost"] = float(cost.mean())
        return stats

    def run(self, is_check=False):
        """运行信号评估；绝不根据评价区间结果自动改变方向。"""
        self._scale()
        if self.resampling_win <= 1:
            print("WARINING: resampling_win:{0}".format(self.resampling_win))

        is_on_mark = (self.factor_data.index.get_level_values(level=0).minute %
                      self.resampling_win == 0)
        self.resample_data = self.factor_data[is_on_mark].copy()

        ic_stats = self.cal_ic()
        if self.resample_data["f_scaled"].dropna().empty:
            raise ValueError("重采样后没有可评价的有效信号")

        pnl_stats = self.cal_pnl()
        autocorr_stats = self._cal_autocorr()
        pnl_stats.update(ic_stats)
        pnl_stats.update(autocorr_stats)

        self.stats = pnl_stats
        if is_check:
            self._check_warnings()
        return self.stats

    def _generate_stats_text(self):
        """在父类摘要后增加信号专用指标。"""
        text = super(FactorEvaluate1, self)._generate_stats_text()
        signal_text = (
            "\n--- Signal Characteristics ---\n"
            f"{'Signal Rate':<25}: {self.stats['signal_rate']:.2%}\n"
            f"{'Flat Rate':<25}: {self.stats['flat_rate']:.2%}\n"
            f"{'Long Rate':<25}: {self.stats['long_rate']:.2%}\n"
            f"{'Short Rate':<25}: {self.stats['short_rate']:.2%}\n"
            f"{'Active Win Rate':<25}: "
            f"{self.stats['active_win_rate']:.2%}\n"
            f"{'Long Win Rate':<25}: {self.stats['long_win_rate']:.2%}\n"
            f"{'Short Win Rate':<25}: {self.stats['short_win_rate']:.2%}\n"
            f"{'Avg Abs Position':<25}: "
            f"{self.stats['avg_abs_position']:.4f}\n"
            f"{'Total Cost':<25}: {self.stats['total_cost']:.6f}\n")
        return text + signal_text

    def plot_results(self):
        """保留父类完整图，并额外绘制信号结构诊断图。"""
        super(FactorEvaluate1, self).plot_results()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(f"Signal Evaluation: {self.code} | {self.factor_name}",
                     fontsize=16)

        rates = pd.Series({
            "Long": self.stats["long_rate"],
            "Short": self.stats["short_rate"],
            "Flat": self.stats["flat_rate"],
        })
        rates.plot.bar(ax=axes[0], color=["#d95f02", "#1b9e77", "#7570b3"])
        axes[0].set_title("Position Distribution")
        axes[0].set_ylabel("Probability")
        axes[0].set_ylim(0, max(1.0, float(rates.max()) * 1.15))
        axes[0].tick_params(axis="x", rotation=0)
        for index, value in enumerate(rates):
            axes[0].text(index,
                         value,
                         f"{value:.1%}",
                         ha="center",
                         va="bottom")

        win_rates = pd.Series({
            "Active": self.stats["active_win_rate"],
            "Long": self.stats["long_win_rate"],
            "Short": self.stats["short_win_rate"],
        })
        win_rates.plot.bar(ax=axes[1], color=["#4c78a8", "#d95f02", "#1b9e77"])
        axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axes[1].set_title("Win Rate by Position")
        axes[1].set_ylabel("Win Rate")
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis="x", rotation=0)
        for index, value in enumerate(win_rates):
            if np.isfinite(value):
                axes[1].text(index,
                             value,
                             f"{value:.1%}",
                             ha="center",
                             va="bottom")

        avg_returns = pd.Series({
            "Long": self.stats["long_avg_ret"] * 10000,
            "Short": self.stats["short_avg_ret"] * 10000,
        })
        colors = [
            "#d95f02" if value >= 0 else "#b2182b" for value in avg_returns
        ]
        avg_returns.plot.bar(ax=axes[2], color=colors)
        axes[2].axhline(0, color="black", linewidth=1)
        axes[2].set_title("Average Net Return by Position")
        axes[2].set_ylabel("Basis Points")
        axes[2].tick_params(axis="x", rotation=0)
        for index, value in enumerate(avg_returns):
            axes[2].text(index,
                         value,
                         f"{value:.3f}",
                         ha="center",
                         va="bottom" if value >= 0 else "top")

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()
        self.signal_figure = fig

    def save_results(self, base_output_dir: str):
        """保存父类结果，并追加信号结构诊断图。"""
        super(FactorEvaluate1, self).save_results(base_output_dir)
        if not hasattr(self, "signal_figure"):
            raise RuntimeError("Please run the 'plot_results()' method first.")

        output_dir = os.path.join(base_output_dir, str(self.name))
        signal_path = os.path.join(output_dir, "signal_evaluation_plot.png")
        self.signal_figure.savefig(signal_path, dpi=150, bbox_inches="tight")
        plt.close(self.signal_figure)
        print(f"Signal evaluation plot saved to: {signal_path}")
