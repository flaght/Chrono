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
        short_mask = position < -position_epsilon
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
        if stats["long_count"] + stats["short_count"] + stats[
                "flat_count"] != len(position):
            raise RuntimeError("多头、空头、空仓样本数与总样本数不一致")
        if not np.isclose(stats["long_rate"] + stats["short_rate"] +
                          stats["flat_rate"], 1.0, rtol=0, atol=1e-10):
            raise RuntimeError("多头、空头、空仓比例口径不一致")
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
        """在一张主图中同时展示收益、IC和信号专用诊断。"""
        if self.stats is None:
            raise RuntimeError("Please run the 'run()' method before plotting.")

        def set_sequential_xticks(ax, series, num_ticks=7):
            if len(series) == 0:
                return
            positions = np.linspace(0, len(series) - 1, num_ticks,
                                    dtype=int)
            labels = [series.index[index].strftime("%Y-%m-%d")
                      for index in positions]
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right")

        fig, axes = plt.subplots(4, 2, figsize=(18, 21))
        fig.suptitle(
            f"Signal Evaluation: {self.code} | {self.expression}",
            fontsize=18)

        nav = self.resample_data["nav"].dropna()
        gross_nav = (1 + self.resample_data["gross_ret"]).cumprod().dropna()
        nav.plot(ax=axes[0, 0], color="blue", label="Net Asset Value",
                 use_index=False)
        gross_nav.plot(ax=axes[0, 0], color="orange", linestyle="--",
                       label="Cumulative Gross Return", use_index=False)
        set_sequential_xticks(axes[0, 0], nav)
        axes[0, 0].set_title("Performance")
        axes[0, 0].legend()

        axes[0, 1].axis("off")
        summary = (
            f"--- Core Performance ---\n"
            f"Annualized Sharpe : {self.stats['sharpe2']:.3f}\n"
            f"Calmar Ratio      : {self.stats['calmar']:.3f}\n"
            f"Active Win Rate   : {self.stats['win_rate']:.2%}\n"
            f"Profit/Loss Ratio : {self.stats['profit_ratio']:.3f}\n"
            f"Total Return      : {self.stats['total_ret']:.2%}\n"
            f"Max Drawdown      : {self.stats['max_dd']:.2%}\n"
            f"Mean Turnover     : {self.stats['turnover']:.4f}\n"
            f"\n--- Signal Structure ---\n"
            f"Signal Rate       : {self.stats['signal_rate']:.2%}\n"
            f"Long Rate         : {self.stats['long_rate']:.2%}\n"
            f"Short Rate        : {self.stats['short_rate']:.2%}\n"
            f"Flat Rate         : {self.stats['flat_rate']:.2%}\n"
            f"Long Win Rate     : {self.stats['long_win_rate']:.2%}\n"
            f"Short Win Rate    : {self.stats['short_win_rate']:.2%}\n"
            f"Average Position  : {self.stats['avg_abs_position']:.4f}\n"
            f"Total Cost        : {self.stats['total_cost']:.6f}")
        axes[0, 1].text(0.05, 0.95, summary, va="top",
                        fontfamily="monospace", fontsize=12)
        axes[0, 1].set_title("Key Performance and Signal Indicators")

        ic = self.resample_data["ic"].dropna()
        cumulative_ic = self.resample_data["cumsum_ic"].dropna()
        ic.plot(ax=axes[1, 0], color="steelblue", alpha=0.7,
                label="Rolling IC", use_index=False)
        set_sequential_xticks(axes[1, 0], ic)
        ic_axis = axes[1, 0].twinx()
        cumulative_ic.plot(ax=ic_axis, color="black", linestyle="--",
                           label="Cumulative IC", use_index=False)
        axes[1, 0].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[1, 0].set_title("IC Analysis")

        axes[1, 1].scatter(self.resample_data["f_scaled"],
                           self.resample_data[self.ret_name], s=8,
                           alpha=0.25, color="purple")
        axes[1, 1].set_title("Signal vs. Forward Return")
        axes[1, 1].set_xlabel("Position")
        axes[1, 1].set_ylabel("Forward Return")

        drawdown = ((self.resample_data["nav"] /
                     self.resample_data["nav"].cummax() - 1) * 100).dropna()
        drawdown.plot(ax=axes[2, 0], color="red", use_index=False)
        axes[2, 0].fill_between(np.arange(len(drawdown)), drawdown.values, 0,
                                color="red", alpha=0.2)
        set_sequential_xticks(axes[2, 0], drawdown)
        axes[2, 0].set_title(
            f"Drawdown (Max = {self.stats['max_dd']:.2%})")
        axes[2, 0].set_ylabel("Drawdown (%)")

        turnover = self.resample_data["turnover"].dropna()
        turnover.plot(ax=axes[2, 1], color="teal", use_index=False)
        set_sequential_xticks(axes[2, 1], turnover)
        axes[2, 1].set_title(
            f"Turnover (Mean = {self.stats['turnover']:.4f})")

        rates = pd.Series({"Long": self.stats["long_rate"],
                           "Short": self.stats["short_rate"],
                           "Flat": self.stats["flat_rate"]})
        rates.plot.bar(ax=axes[3, 0],
                       color=["#d95f02", "#1b9e77", "#7570b3"])
        axes[3, 0].set_title("Position Distribution")
        axes[3, 0].set_ylabel("Probability")
        axes[3, 0].set_ylim(0, 1)
        axes[3, 0].tick_params(axis="x", rotation=0)
        for index, value in enumerate(rates):
            axes[3, 0].text(index, value, f"{value:.1%}", ha="center",
                            va="bottom")

        win_rates = pd.Series({"Active": self.stats["active_win_rate"],
                               "Long": self.stats["long_win_rate"],
                               "Short": self.stats["short_win_rate"]})
        win_rates.plot.bar(ax=axes[3, 1],
                           color=["#4c78a8", "#d95f02", "#1b9e77"])
        axes[3, 1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axes[3, 1].set_title("Win Rate by Position")
        axes[3, 1].set_ylabel("Win Rate")
        axes[3, 1].set_ylim(0, 1)
        axes[3, 1].tick_params(axis="x", rotation=0)
        for index, value in enumerate(win_rates):
            if np.isfinite(value):
                axes[3, 1].text(index, value, f"{value:.1%}", ha="center",
                                va="bottom")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        plt.show()
        self.figure = fig

    def save_results(self, base_output_dir: str):
        """保存合并后的单张主图及父类全部结果。"""
        super(FactorEvaluate1, self).save_results(base_output_dir)
