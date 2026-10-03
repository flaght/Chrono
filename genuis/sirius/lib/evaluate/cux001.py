"""Polars implementation of the time-series factor & signal evaluator.

保持纯 Polars 的极致高性能计算路径，全面融合并支持 cux004 的实盘信号评价体系：
1. 强制单品种检查与过滤，杜绝跨品种串联。
2. 修复胜率口径为“有仓位时点胜率”（active_win_rate），同时保留全时点胜率（time_win_rate）。
3. 精确区分 profit_factor（利润因子：总盈/总亏绝对值）与 payoff_ratio（真实单笔盈亏比：均笔盈利/均笔亏损绝对值）。
4. 全面统计多头/空头/空仓比例、样本数、胜率、均次收益、换手率与交易成本。
5. 严格保证样本数一致性（long_count + short_count + flat_count == total）与比例和为 1 的口径校验。
6. 默认 auto=False 禁止依据评价期 IC 自动反转信号（杜绝未来信息泄漏）。
7. 支持 lag_signal 防前视时序对齐选项与 [-1, 1] 仓位防呆截断。
8. 完整的 4x2 综合诊断大图（纯 Matplotlib 绘制，无 Seaborn 依赖）。
"""

import os
from xml.dom import minidom
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


def _rolling_minimum_samples(value: int):
    """支持 Polars 1.21 min_periods -> min_samples 重命名."""
    parts = pl.__version__.split('.')
    try:
        version = tuple(int(part.split('-')[0]) for part in parts[:2])
    except ValueError:
        version = (1, 21)
    key = 'min_samples' if version >= (1, 21) else 'min_periods'
    return {key: value}


class FactorEvaluate1:
    """高性能纯 Polars 时序因子与交易信号综合评估器."""

    def __init__(self,
                 factor_data,
                 resampling_win: int = 1,
                 factor_name: str = "signal",
                 code: str = "",
                 ret_name: str = "ret",
                 roll_win: int = 252,
                 fee: float = 0.0003,
                 scale_method: str = "raw",
                 annualization_factor: int = 252,
                 expression=None,
                 name=None,
                 auto: bool = False,
                 lag_signal: bool = False):
        self.code = str(code) if code else ""
        self.factor_name = factor_name
        self.ret_name = ret_name
        self.roll_win = int(roll_win)
        self.fee = float(fee)
        self.scale_method = scale_method
        self.annualization_factor = int(annualization_factor)
        self.name = str(name) if name is not None else factor_name
        self.expression = str(expression) if expression is not None else factor_name
        self.auto = bool(auto)
        self.lag_signal = bool(lag_signal)
        self.resampling_win = max(1, int(resampling_win))
        self.stats = None
        self.figure = None
        self.direction_inverted = False
        self.resample_data_pl = None
        self.resample_data = None
        self.factor_data = self._preprocess_data(factor_data)

    def _preprocess_data(self, data) -> pl.DataFrame:
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
        elif isinstance(data, pl.LazyFrame):
            data = data.collect()
        elif not isinstance(data, pl.DataFrame):
            raise TypeError("factor_data must be pandas/polars DataFrame or Polars LazyFrame")

        required = {"trade_time", self.factor_name, self.ret_name}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # 1. 单品种检查与自动过滤
        if "code" in data.columns:
            if self.code:
                data = data.filter(pl.col("code") == self.code)
                if data.is_empty():
                    raise ValueError(f"指定品种 code='{self.code}' 过滤后数据为空")
            unique_codes = data.select(pl.col("code").drop_nulls().unique()).to_series().to_list()
            if len(unique_codes) > 1:
                raise ValueError(
                    f"FactorEvaluate1 仅支持单品种时序评估，检测到多个品种: {unique_codes}。"
                    "请传入 code 参数或在输入数据中先过滤单品种。"
                )
            elif len(unique_codes) == 1 and not self.code:
                self.code = str(unique_codes[0])

        # 2. 规范化 trade_time
        time_dtype = data.schema["trade_time"]
        if time_dtype == pl.String:
            time_expr = pl.col("trade_time").str.to_datetime(strict=False)
        elif time_dtype == pl.Date:
            time_expr = pl.col("trade_time").cast(pl.Datetime)
        else:
            time_expr = pl.col("trade_time").cast(pl.Datetime, strict=False)

        clean_data = (
            data.lazy()
            .select(
                time_expr.alias("trade_time"),
                pl.col(self.factor_name).cast(pl.Float64, strict=False),
                pl.col(self.ret_name).cast(pl.Float64, strict=False),
            )
            .drop_nulls("trade_time")
            .filter(pl.col(self.factor_name).is_finite() & pl.col(self.ret_name).is_finite())
            .sort("trade_time")
            .collect()
        )

        # 3. 时间戳重复性检查 (防止数据重复导致收益/换手串联紊乱)
        if clean_data.select(pl.col("trade_time").is_duplicated().any()).item():
            dup_count = clean_data.select(pl.col("trade_time").is_duplicated().sum()).item()
            print(f"WARNING: 检测到 {dup_count} 行重复的 trade_time 时间戳，自动保留首个观测值去重。")
            clean_data = clean_data.unique(subset=["trade_time"], keep="first").sort("trade_time")

        return clean_data

    def _scaled_frame(self) -> pl.DataFrame:
        x = pl.col(self.factor_name)
        win = self.roll_win

        if self.scale_method == "roll_min_max":
            lo = x.rolling_min(window_size=win)
            hi = x.rolling_max(window_size=win)
            denominator = (hi - lo).clip(lower_bound=1e-8)
            scaled = (2.0 * (x - lo) / denominator - 1.0).clip(lower_bound=-1.0, upper_bound=1.0)
        elif self.scale_method == "roll_zscore":
            mean = x.rolling_mean(window_size=win)
            std = x.rolling_std(window_size=win).clip(lower_bound=1e-8)
            scaled = ((x - mean) / std).clip(lower_bound=-3.0, upper_bound=3.0) / 3.0
        elif self.scale_method == "roll_quantile":
            q25 = x.rolling_quantile(0.25, interpolation="linear", window_size=win)
            q75 = x.rolling_quantile(0.75, interpolation="linear", window_size=win)
            scaled = (2.0 * (x - q25) / (q75 - q25).clip(lower_bound=1e-8) - 1.0).clip(lower_bound=-1.0, upper_bound=1.0)
        elif self.scale_method == "ew_zscore":
            mean = x.ewm_mean(span=win, adjust=False)
            variance = x.ewm_var(span=win, adjust=False)
            scaled = ((x - mean) / variance.sqrt().clip(lower_bound=1e-8)).clip(lower_bound=-3.0, upper_bound=3.0) / 3.0
        elif self.scale_method == "train_const":
            training = self.factor_data.get_column(self.factor_name).head(win)
            mean = training.mean()
            std = training.std()
            std = max(std, 1e-8) if std is not None and np.isfinite(std) else 1e-8
            scaled = ((x - mean) / std).clip(lower_bound=-3.0, upper_bound=3.0) / 3.0
        elif self.scale_method == "raw":
            # 原始仓位/信号截断在 [-1, 1] 之间，防止超杠杆破坏资金管理
            scaled = x.clip(lower_bound=-1.0, upper_bound=1.0)
        else:
            raise ValueError(f"Unknown scale_method: {self.scale_method}")

        return self.factor_data.with_columns(scaled.alias("f_scaled"))

    @staticmethod
    def _safe_ratio(numerator, denominator, default=0.0):
        if denominator is None or not np.isfinite(denominator) or denominator == 0:
            return default
        if numerator is None or not np.isfinite(numerator):
            return default
        return numerator / denominator

    @staticmethod
    def _safe_mean(mask, values):
        """兼容 cux004 的安全均值计算 helper."""
        if hasattr(mask, "any") and not mask.any():
            return np.nan
        if hasattr(values, "loc"):
            return float(values.loc[mask].mean())
        return np.nan

    def cal_ic(self, frame: pl.DataFrame = None) -> tuple[pl.DataFrame, dict]:
        """计算滚动 IC 与全样本 IC 指标."""
        if frame is None:
            frame = self.resample_data_pl
        if frame is None:
            raise RuntimeError("Data frame is not initialized.")

        frame = frame.with_columns(
            pl.rolling_corr(
                pl.col(self.ret_name),
                pl.col(self.factor_name),
                window_size=self.roll_win,
                **_rolling_minimum_samples(5),
            ).alias("ic")
        ).with_columns(pl.col("ic").cum_sum().alias("cumsum_ic"))

        ic_values = frame.select(
            pl.corr(self.ret_name, self.factor_name).alias("total_ic"),
            pl.col("ic").mean().alias("ic_mean"),
            pl.col("ic").std().alias("ic_std"),
        ).row(0, named=True)

        total_ic = ic_values["total_ic"] if ic_values["total_ic"] is not None else np.nan
        ic_mean = ic_values["ic_mean"] if ic_values["ic_mean"] is not None else np.nan
        ic_std = ic_values["ic_std"] if ic_values["ic_std"] is not None else np.nan
        ic_ir = self._safe_ratio(ic_mean, ic_std)

        # auto 控制是否在 IC<0 时反转信号（默认 auto=False，禁止前视改变方向）
        self.direction_inverted = bool(np.isfinite(ic_mean) and ic_mean < 0 and self.auto)
        direction = -1.0 if self.direction_inverted else 1.0

        ic_stats = {
            "total_ic": total_ic,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "direction_inverted": self.direction_inverted,
            "effective_total_ic": total_ic * direction,
            "effective_ic_mean": ic_mean * direction,
            "effective_ic_std": ic_std,
            "effective_ic_ir": ic_ir * direction,
        }
        return frame, ic_stats

    def cal_pnl(self, position_epsilon: float = 1e-12, frame: pl.DataFrame = None) -> tuple[pl.DataFrame, dict]:
        """纯 Polars 高性能单 pass 聚合计算信号回测指标与结构统计."""
        if frame is None:
            frame = self.resample_data_pl
        if frame is None:
            raise RuntimeError("Data frame is not initialized.")

        direction = -1.0 if self.direction_inverted else 1.0

        # 1. 仓位构造与交易成本/净值计算
        pos_expr = pl.col("f_scaled") * direction
        if self.lag_signal:
            pos_expr = pos_expr.shift(1).fill_null(0.0)

        frame = (
            frame.lazy()
            .with_columns(pos_expr.alias("pos"))
            .with_columns(
                (pl.col("pos") * pl.col(self.ret_name)).alias("gross_ret"),
                pl.col("pos").diff().fill_null(pl.col("pos").abs()).abs().alias("turnover"),
            )
            .with_columns(
                (pl.col("gross_ret") - self.fee * pl.col("turnover")).alias("net_ret")
            )
            .with_columns(
                (pl.lit(1.0) + pl.col("net_ret")).cum_prod().alias("nav"),
                (pl.lit(1.0) + pl.col("gross_ret")).cum_prod().alias("gross_nav"),
            )
            .collect()
        )

        total_obs = len(frame)
        eps = position_epsilon
        active_expr = pl.col("pos").abs() > eps
        long_expr = pl.col("pos") > eps
        short_expr = pl.col("pos") < -eps
        flat_expr = pl.col("pos").abs() <= eps

        pos_ret_expr = pl.col("net_ret") > 0
        neg_ret_expr = pl.col("net_ret") < 0

        # 单 pass 向量化聚合
        signal_agg = frame.select([
            # 仓位覆盖结构
            active_expr.mean().alias("signal_rate"),
            flat_expr.mean().alias("flat_rate"),
            long_expr.mean().alias("long_rate"),
            short_expr.mean().alias("short_rate"),
            pl.col("pos").abs().mean().alias("avg_abs_position"),
            long_expr.sum().alias("long_count"),
            short_expr.sum().alias("short_count"),
            flat_expr.sum().alias("flat_count"),
            # 胜率口径：有仓位胜率 vs 全时点胜率
            pos_ret_expr.mean().alias("time_win_rate"),
            pl.when(active_expr).then(pos_ret_expr).otherwise(None).mean().alias("active_win_rate"),
            pl.when(long_expr).then(pos_ret_expr).otherwise(None).mean().alias("long_win_rate"),
            pl.when(short_expr).then(pos_ret_expr).otherwise(None).mean().alias("short_win_rate"),
            # 多空收益拆分
            pl.when(long_expr).then(pl.col("net_ret")).otherwise(None).mean().alias("long_avg_ret"),
            pl.when(short_expr).then(pl.col("net_ret")).otherwise(None).mean().alias("short_avg_ret"),
            pl.when(long_expr).then(pl.col("net_ret")).otherwise(0.0).sum().alias("long_total_ret"),
            pl.when(short_expr).then(pl.col("net_ret")).otherwise(0.0).sum().alias("short_total_ret"),
            # 总收益与费用
            pl.col("gross_ret").sum().alias("gross_ret_sum"),
            pl.col("net_ret").sum().alias("net_ret_sum"),
            (self.fee * pl.col("turnover")).sum().alias("total_cost"),
            (self.fee * pl.col("turnover")).mean().alias("avg_cost"),
            # 盈亏比拆分指标
            pl.when(pos_ret_expr).then(pl.col("net_ret")).otherwise(0.0).sum().alias("pos_sum"),
            pl.when(neg_ret_expr).then(pl.col("net_ret").abs()).otherwise(0.0).sum().alias("neg_abs_sum"),
            pl.when(pos_ret_expr).then(pl.col("net_ret")).otherwise(None).mean().alias("pos_mean"),
            pl.when(neg_ret_expr).then(pl.col("net_ret").abs()).otherwise(None).mean().alias("neg_abs_mean"),
        ]).row(0, named=True)

        clean_signal_stats = {}
        for k, v in signal_agg.items():
            if v is None:
                if k.endswith("_count"):
                    clean_signal_stats[k] = 0
                elif k.endswith("_total_ret") or k.endswith("_sum") or k == "total_cost":
                    clean_signal_stats[k] = 0.0
                else:
                    clean_signal_stats[k] = np.nan
            elif isinstance(v, (int, np.integer)):
                clean_signal_stats[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                clean_signal_stats[k] = float(v)
            else:
                clean_signal_stats[k] = v

        # cux004 严格一致性校验
        if clean_signal_stats["long_count"] + clean_signal_stats["short_count"] + clean_signal_stats["flat_count"] != total_obs:
            raise RuntimeError("多头、空头、空仓样本数与总样本数不一致")
        if not np.isclose(clean_signal_stats["long_rate"] + clean_signal_stats["short_rate"] + clean_signal_stats["flat_rate"], 1.0, rtol=0, atol=1e-10):
            raise RuntimeError("多头、空头、空仓比例口径不一致")

        # 利润因子 (总盈 / 总亏绝对值) 与 真实单笔盈亏比 (均笔盈利 / 均笔亏损绝对值)
        pos_sum = clean_signal_stats["pos_sum"]
        neg_abs_sum = clean_signal_stats["neg_abs_sum"]
        profit_factor = pos_sum / neg_abs_sum if neg_abs_sum > 0 else (np.inf if pos_sum > 0 else 0.0)

        pos_mean = clean_signal_stats["pos_mean"]
        neg_abs_mean = clean_signal_stats["neg_abs_mean"]
        payoff_ratio = pos_mean / neg_abs_mean if (np.isfinite(neg_abs_mean) and neg_abs_mean > 0) else (np.inf if np.isfinite(pos_mean) and pos_mean > 0 else 0.0)

        # 复合日度年化夏普与卡玛比率
        daily = (
            frame.lazy()
            .with_columns(pl.col("trade_time").dt.date().alias("_date"))
            .group_by("_date")
            .agg(((pl.col("net_ret") + 1.0).product() - 1.0).alias("daily_ret"))
            .sort("_date")
            .collect()
        )
        daily_mean = daily.get_column("daily_ret").mean()
        daily_std = daily.get_column("daily_ret").std()
        annualized_sharpe = self._safe_ratio(
            daily_mean * self.annualization_factor,
            daily_std * np.sqrt(self.annualization_factor)
        )

        nav_series = frame.get_column("nav")
        total_ret = float(nav_series[-1] - 1.0)
        elapsed_days = (frame["trade_time"][-1] - frame["trade_time"][0]).days
        years = max(elapsed_days / 365.0, 1e-6)
        annualized_return = (1.0 + total_ret) ** (1.0 / years) - 1.0

        max_dd = float(frame.select(
            (pl.col("nav") / pl.col("nav").cum_max().clip(lower_bound=1.0) - 1.0).min()
        ).item())

        period_mean = float(frame.get_column("net_ret").mean())
        period_std = float(frame.get_column("net_ret").std())

        # 因子统计特征
        factor_col = pl.col(self.factor_name)
        ret_col = pl.col(self.ret_name)
        distribution = frame.select(
            factor_col.is_not_null().mean().alias("factor_coverage"),
            pl.col("f_scaled").is_not_null().mean().alias("signal_coverage"),
            factor_col.mean().alias("factor_mean"),
            factor_col.std().alias("factor_std"),
            factor_col.skew(bias=False).alias("factor_skew"),
            factor_col.kurtosis(fisher=True, bias=False).alias("factor_kurtosis"),
            pl.corr(factor_col, factor_col.shift(1)).alias("factor_autocorr"),
            pl.corr(ret_col, ret_col.shift(1)).alias("ret_autocorr"),
        ).row(0, named=True)
        distribution = {k: float(v) if (v is not None and np.isfinite(v)) else np.nan for k, v in distribution.items()}

        pnl_stats = {
            "total_ret": total_ret,
            "avg_ret": period_mean,
            "max_dd": max_dd,
            "calmar": annualized_return / abs(max_dd) if max_dd != 0 else np.nan,
            "sharpe1": self._safe_ratio(period_mean, period_std),
            "sharpe2": annualized_sharpe,
            "turnover": float(frame.get_column("turnover").mean()),
            # win_rate 统一定义为有仓位胜率，同时保留 time_win_rate 全时点胜率
            "win_rate": clean_signal_stats["active_win_rate"],
            "profit_factor": profit_factor,
            "payoff_ratio": payoff_ratio,
            "profit_ratio": profit_factor,  # 兼容历史别名
            **clean_signal_stats,
            **distribution,
            "code": self.code,
        }
        return frame, pnl_stats

    def run(self, is_check: bool = False) -> dict:
        """执行端到端因子/信号评估."""
        if self.resampling_win <= 0:
            raise ValueError("resampling_win must be a positive integer")

        if self.resampling_win <= 1:
            print(f"WARNING: resampling_win:{self.resampling_win}")

        # 1. 滚动缩放与重采样过滤
        frame = (
            self._scaled_frame().lazy()
            .filter(pl.col("trade_time").dt.minute() % self.resampling_win == 0)
            .drop_nulls([self.ret_name, "f_scaled"])
            .collect()
        )
        if frame.is_empty():
            raise ValueError("重采样后没有可评价的有效观测值。")

        # 2. IC 计算
        frame, ic_stats = self.cal_ic(frame)

        # 3. PnL 与信号指标计算
        frame, pnl_stats = self.cal_pnl(frame=frame)

        pnl_stats.update(ic_stats)
        self.stats = pnl_stats
        self.resample_data_pl = frame
        self.resample_data = None

        if is_check:
            self._check_warnings()
        return self.stats

    def _check_warnings(self):
        """输出合理性检查与警告."""
        print("\n--- Sanity Checks & Warnings ---")
        ret_ac = self.stats.get("ret_autocorr", np.nan)
        if not np.isfinite(ret_ac) or not (-0.1 < ret_ac < 0.1):
            print(f"⚠️  WARNING: 收益率自相关性为 {ret_ac:.3f}，脱离正常范围 [-0.1, 0.1]。")
        else:
            print(f"✅ 收益率自相关性 ({ret_ac:.3f}) 正常。")

        factor_ac = self.stats.get("factor_autocorr", np.nan)
        if np.isfinite(factor_ac) and abs(factor_ac) > 0.99:
            print(f"⚠️  WARNING: 因子自相关性过高 ({factor_ac:.3f})，信号接近非平稳。")
        else:
            print(f"✅ 因子自相关性 ({factor_ac:.3f}) 处于合理区间。")

        ic_ir = self.stats.get("ic_ir", np.nan)
        if not np.isfinite(ic_ir) or ic_ir < 0.3:
            print(f"⚠️  WARNING: ICIR 为 {ic_ir:.3f}，预测稳定性较弱。")
        else:
            print(f"✅ ICIR ({ic_ir:.3f}) 表现稳定。")

        if self.stats.get("signal_rate", 0) < 0.05:
            print(f"⚠️  WARNING: 有效交易信号覆盖率过低 ({self.stats['signal_rate']:.2%})，可能存在样本不足。")

    def _generate_stats_text(self) -> str:
        """生成兼具 cux001 性能指标与 cux004 信号特性的格式化报告文本."""
        parts = []
        label = self.expression or self.name
        if label:
            parts.append(f"Expression: {label}")
        if self.name and self.name != label:
            parts.append(f"Name: {self.name}")
        if self.code:
            parts.append(f"Code: {self.code}")
        if parts:
            parts.append("")

        def fmt_pct(val):
            return f"{val:.2%}" if (val is not None and np.isfinite(val)) else "N/A"

        def fmt_num(val, precision=3):
            return f"{val:.{precision}f}" if (val is not None and np.isfinite(val)) else "N/A"

        def fmt_bps(val):
            return f"{val * 10000:.2f} bps" if (val is not None and np.isfinite(val)) else "N/A"

        parts.append(
            "--- Core Performance ---\n"
            f"{'Annualized Sharpe':<25}: {fmt_num(self.stats.get('sharpe2'))}\n"
            f"{'Calmar Ratio':<25}: {fmt_num(self.stats.get('calmar'))}\n"
            f"{'Active Win Rate':<25}: {fmt_pct(self.stats.get('win_rate'))}\n"
            f"{'Time Win Rate':<25}: {fmt_pct(self.stats.get('time_win_rate'))}\n"
            f"{'Profit Factor':<25}: {fmt_num(self.stats.get('profit_factor'))}\n"
            f"{'Payoff Ratio':<25}: {fmt_num(self.stats.get('payoff_ratio'))}\n"
            f"{'Total Return':<25}: {fmt_pct(self.stats.get('total_ret'))}\n"
            f"{'Max Drawdown':<25}: {fmt_pct(self.stats.get('max_dd'))}\n"
            f"{'Mean Turnover':<25}: {fmt_num(self.stats.get('turnover'), 4)}\n"
            f"{'Total Cost':<25}: {fmt_num(self.stats.get('total_cost'), 6)}\n"
            "\n--- Signal Characteristics ---\n"
            f"{'Signal Rate':<25}: {fmt_pct(self.stats.get('signal_rate'))}\n"
            f"{'Flat Rate':<25}: {fmt_pct(self.stats.get('flat_rate'))}\n"
            f"{'Long Rate':<25}: {fmt_pct(self.stats.get('long_rate'))}\n"
            f"{'Short Rate':<25}: {fmt_pct(self.stats.get('short_rate'))}\n"
            f"{'Long Win Rate':<25}: {fmt_pct(self.stats.get('long_win_rate'))}\n"
            f"{'Short Win Rate':<25}: {fmt_pct(self.stats.get('short_win_rate'))}\n"
            f"{'Long Avg Return':<25}: {fmt_bps(self.stats.get('long_avg_ret'))}\n"
            f"{'Short Avg Return':<25}: {fmt_bps(self.stats.get('short_avg_ret'))}\n"
            f"{'Avg Abs Position':<25}: {fmt_num(self.stats.get('avg_abs_position'), 4)}\n"
            "\n--- Factor & IC Metrics ---\n"
            f"{'Total IC':<25}: {fmt_num(self.stats.get('total_ic'), 4)}\n"
            f"{'IC Mean':<25}: {fmt_num(self.stats.get('ic_mean'), 4)}\n"
            f"{'ICIR':<25}: {fmt_num(self.stats.get('ic_ir'), 4)}\n"
            f"{'Factor Autocorr':<25}: {fmt_num(self.stats.get('factor_autocorr'), 4)}\n"
            f"{'Return Autocorr':<25}: {fmt_num(self.stats.get('ret_autocorr'), 4)}\n"
        )
        return "\n".join(parts)

    def to_pandas(self) -> pd.DataFrame:
        """按需将 Polars 结果转为带 trade_time 索引的 Pandas DataFrame."""
        if self.resample_data_pl is None:
            raise RuntimeError("Please run the 'run()' method first.")
        if self.resample_data is None:
            self.resample_data = self.resample_data_pl.to_pandas().set_index("trade_time")
        return self.resample_data

    def plot_results(self):
        """完整实现 cux004 的 4x2 单张综合评估诊断图表（纯 Matplotlib，无 Seaborn 依赖）."""
        if self.stats is None:
            raise RuntimeError("Please run the 'run()' method before plotting.")
        data = self.to_pandas()

        def set_sequential_xticks(ax, series, num_ticks=7):
            if len(series) == 0:
                return
            positions = np.linspace(0, len(series) - 1, num_ticks, dtype=int)
            labels = [series.index[idx].strftime("%Y-%m-%d") for idx in positions]
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right")

        fig, axes = plt.subplots(4, 2, figsize=(18, 21))
        label = self.expression or self.name
        title = f"Signal Evaluation: {self.code} | {label}" if self.code else f"Signal Evaluation: {label}"
        fig.suptitle(title, fontsize=18)

        # 1. 净值与累计毛收益 (Performance)
        nav = data["nav"].dropna()
        gross_nav = data["gross_nav"].dropna()
        nav.plot(ax=axes[0, 0], color="blue", label="Net Asset Value", use_index=False)
        gross_nav.plot(ax=axes[0, 0], color="orange", linestyle="--", label="Cumulative Gross Return", use_index=False)
        set_sequential_xticks(axes[0, 0], nav)
        axes[0, 0].set_title("Performance")
        axes[0, 0].set_ylabel("NAV")
        axes[0, 0].legend()

        # 2. 核心表现与信号结构指示卡片 (Key Indicators)
        axes[0, 1].axis("off")
        summary = (
            f"--- Core Performance ---\n"
            f"Annualized Sharpe : {self.stats['sharpe2']:.3f}\n"
            f"Calmar Ratio      : {self.stats['calmar']:.3f}\n"
            f"Active Win Rate   : {self.stats['win_rate']:.2%}\n"
            f"Profit Factor     : {self.stats['profit_factor']:.3f}\n"
            f"Payoff Ratio      : {self.stats['payoff_ratio']:.3f}\n"
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
            f"Total Cost        : {self.stats['total_cost']:.6f}"
        )
        axes[0, 1].text(0.05, 0.95, summary, va="top", fontfamily="monospace", fontsize=12)
        axes[0, 1].set_title("Key Performance and Signal Indicators")

        # 3. 滚动 IC 与累计 IC 分析 (IC Analysis)
        ic = data["ic"].dropna()
        cumulative_ic = data["cumsum_ic"].dropna()
        ic.plot(ax=axes[1, 0], color="steelblue", alpha=0.7, label="Rolling IC", use_index=False)
        set_sequential_xticks(axes[1, 0], ic)
        ic_axis = axes[1, 0].twinx()
        cumulative_ic.plot(ax=ic_axis, color="black", linestyle="--", label="Cumulative IC", use_index=False)
        axes[1, 0].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[1, 0].set_title("IC Analysis")
        axes[1, 0].set_ylabel("Rolling IC", color="steelblue")
        ic_axis.set_ylabel("Cumulative IC", color="black")

        # 4. 信号 vs 未来收益散点图 (Signal vs. Forward Return)
        axes[1, 1].scatter(data["pos"], data[self.ret_name], s=8, alpha=0.25, color="purple")
        axes[1, 1].set_title("Signal vs. Forward Return")
        axes[1, 1].set_xlabel("Position")
        axes[1, 1].set_ylabel("Forward Return")

        # 5. 回撤曲线 (Drawdown)
        drawdown = ((data["nav"] / data["nav"].cummax().clip(lower=1.0) - 1.0) * 100).dropna()
        drawdown.plot(ax=axes[2, 0], color="red", use_index=False)
        axes[2, 0].fill_between(np.arange(len(drawdown)), drawdown.values, 0, color="red", alpha=0.2)
        set_sequential_xticks(axes[2, 0], drawdown)
        axes[2, 0].set_title(f"Drawdown (Max = {self.stats['max_dd']:.2%})")
        axes[2, 0].set_ylabel("Drawdown (%)")

        # 6. 换手率时序 (Turnover)
        turnover = data["turnover"].dropna()
        turnover.plot(ax=axes[2, 1], color="teal", use_index=False)
        set_sequential_xticks(axes[2, 1], turnover)
        axes[2, 1].set_title(f"Turnover (Mean = {self.stats['turnover']:.4f})")
        axes[2, 1].set_ylabel("Turnover")

        # 7. 仓位分布柱状图 (Position Distribution)
        rates = pd.Series({
            "Long": self.stats["long_rate"],
            "Short": self.stats["short_rate"],
            "Flat": self.stats["flat_rate"]
        })
        rates.plot.bar(ax=axes[3, 0], color=["#d95f02", "#1b9e77", "#7570b3"])
        axes[3, 0].set_title("Position Distribution")
        axes[3, 0].set_ylabel("Probability")
        axes[3, 0].set_ylim(0, 1)
        axes[3, 0].tick_params(axis="x", rotation=0)
        for index, value in enumerate(rates):
            if np.isfinite(value):
                axes[3, 0].text(index, value + 0.02, f"{value:.1%}", ha="center", va="bottom")

        # 8. 分仓胜率柱状图 (Win Rate by Position)
        win_rates = pd.Series({
            "Active": self.stats["active_win_rate"],
            "Long": self.stats["long_win_rate"],
            "Short": self.stats["short_win_rate"]
        })
        win_rates.plot.bar(ax=axes[3, 1], color=["#4c78a8", "#d95f02", "#1b9e77"])
        axes[3, 1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axes[3, 1].set_title("Win Rate by Position")
        axes[3, 1].set_ylabel("Win Rate")
        axes[3, 1].set_ylim(0, 1)
        axes[3, 1].tick_params(axis="x", rotation=0)
        for index, value in enumerate(win_rates):
            if np.isfinite(value):
                axes[3, 1].text(index, value + 0.02, f"{value:.1%}", ha="center", va="bottom")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        self.figure = fig
        return fig

    def generate_xml(self, start_time: str, end_time: str) -> str:
        """生成详细评估指标 XML."""
        if self.stats is None:
            raise RuntimeError("Please run the 'run()' method first.")

        def add_text(parent, tag, value, **attributes):
            element = ET.SubElement(parent, tag, {
                key: str(item) for key, item in attributes.items() if item is not None
            })
            if value is None:
                element.set("status", "missing")
            else:
                element.text = str(value)
            return element

        def add_metric(parent, tag, value, description, unit="ratio"):
            element = ET.SubElement(parent, tag, {"description": description, "unit": unit})
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = np.nan
            if np.isfinite(number):
                element.text = format(number, ".12g")
            else:
                element.set("status", "missing")
            return element

        root = ET.Element("factor_evaluation", {"schema_version": "2.0.0", "purpose": "signal_and_factor_screening"})
        identity = ET.SubElement(root, "identity")
        add_text(identity, "name", self.name)
        add_text(identity, "expression", self.expression)
        add_text(identity, "code", self.code)
        add_text(identity, "factor_field", self.factor_name)
        add_text(identity, "forward_return_field", self.ret_name)

        config = ET.SubElement(root, "evaluation_config")
        add_text(config, "start_time", start_time)
        add_text(config, "end_time", end_time)
        add_text(config, "rolling_window", self.roll_win, unit="period")
        add_text(config, "resampling_window", self.resampling_win, unit="minute")
        add_text(config, "scale_method", self.scale_method)
        add_text(config, "annualization_factor", self.annualization_factor, unit="periods_per_year")
        add_text(config, "return_basis", "net_return_after_costs")
        add_text(config, "engine", "polars_high_perf")

        metrics = ET.SubElement(root, "metrics", {"basis": "net_return_after_costs"})
        for tag, key, desc, unit in (
            ("average_return", "avg_ret", "平均收益", "decimal_return"),
            ("total_return", "total_ret", "累计成本后净收益", "decimal_return"),
            ("annualized_sharpe", "sharpe2", "年化夏普", "ratio"),
            ("maximum_drawdown", "max_dd", "最大回撤", "decimal_return"),
            ("calmar_ratio", "calmar", "卡玛比率", "ratio"),
            ("active_win_rate", "active_win_rate", "有仓位胜率", "decimal_ratio"),
            ("time_win_rate", "time_win_rate", "全时点胜率", "decimal_ratio"),
            ("profit_factor", "profit_factor", "利润因子 (总盈/总亏)", "ratio"),
            ("payoff_ratio", "payoff_ratio", "盈亏比 (均笔盈利/均笔亏损)", "ratio"),
            ("turnover", "turnover", "平均换手率", "turnover"),
            ("total_cost", "total_cost", "累计手续费成本", "decimal_cost"),
        ):
            add_metric(metrics, tag, self.stats.get(key), desc, unit)

        signal_elem = ET.SubElement(root, "signal_structure")
        for tag, key, desc in (
            ("signal_rate", "signal_rate", "持仓覆盖率"),
            ("long_rate", "long_rate", "多头持仓比例"),
            ("short_rate", "short_rate", "空头持仓比例"),
            ("flat_rate", "flat_rate", "空仓比例"),
            ("long_win_rate", "long_win_rate", "多头胜率"),
            ("short_win_rate", "short_win_rate", "空头胜率"),
        ):
            add_metric(signal_elem, tag, self.stats.get(key), desc, "decimal_ratio")

        rough_xml = ET.tostring(root, encoding="utf-8")
        return minidom.parseString(rough_xml).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    def save_results(self, base_output_dir: str):
        """持久化评估结果、文本、CSV、单张 4x2 诊断图与 XML."""
        if self.stats is None:
            raise RuntimeError("Please run the 'run()' method before saving.")
        output_dir = os.path.join(base_output_dir, str(self.name))
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "performance_summary.txt"), "w", encoding="utf-8") as file:
            file.write(self._generate_stats_text())

        data = self.to_pandas()
        for metric in ("nav", "ic", "turnover"):
            if metric in data.columns:
                data[metric].to_csv(os.path.join(output_dir, f"{metric}.csv"), header=True)

        if self.figure is None:
            self.plot_results()

        self.figure.savefig(os.path.join(output_dir, "evaluation_plot.png"), dpi=150)
        plot_dir = os.path.join(base_output_dir, "plot")
        os.makedirs(plot_dir, exist_ok=True)
        self.figure.savefig(os.path.join(plot_dir, f"{self.name}.png"), dpi=150)
        plt.close(self.figure)

        xml_text = self.generate_xml(
            self.resample_data_pl["trade_time"][0].isoformat(),
            self.resample_data_pl["trade_time"][-1].isoformat()
        )
        with open(os.path.join(output_dir, "evaluation.xml"), "w", encoding="utf-8") as file:
            file.write(xml_text)
        xml_dir = os.path.join(base_output_dir, "xml")
        os.makedirs(xml_dir, exist_ok=True)
        with open(os.path.join(xml_dir, f"{self.name}.xml"), "w", encoding="utf-8") as file:
            file.write(xml_text)

    def cal_returns(self) -> dict:
        """提供多空收益统计."""
        if self.stats is None:
            raise RuntimeError("Please run the 'run()' method first.")
        return {
            "long_count": self.stats.get("long_count", 0),
            "long_sum_returns": self.stats.get("long_total_ret", 0.0),
            "long_avg_returns": self.stats.get("long_avg_ret", 0.0),
            "long_win_ratio": self.stats.get("long_win_rate", 0.0),
            "short_count": self.stats.get("short_count", 0),
            "short_sum_returns": self.stats.get("short_total_ret", 0.0),
            "short_avg_returns": self.stats.get("short_avg_ret", 0.0),
            "short_win_ratio": self.stats.get("short_win_rate", 0.0),
        }


FactorEvaluatePolars = FactorEvaluate1
__all__ = ["FactorEvaluate1", "FactorEvaluatePolars"]
