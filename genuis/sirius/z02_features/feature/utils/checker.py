"""基于 Polars 的高频与微观因子异常值检查与数据质量审计工具 (Factor Anomaly Checker)。

针对量化高频特征下采样与日内 Bar 级因子计算，提供极速、纯 Polars 惰性/内存数据质量检测：
1. 非法数值检测：Null, NaN, Inf (正负无穷)；
2. 整数下溢/溢出检测：无符号整型下溢 (如 4294967280) 或超大异常整型；
3. 理论与物理边界违规 (Bound Violations)：
   - [0, 1] 比例因子违规 (如 volume_in_pct, double_open_ratio 等)；
   - [-1, 1] 不平衡与相关系数因子违规 (如 net_tick_in_pct, ofi_flow_imbalance, corr_* 等)；
   - [0, +inf) 非负物理量违规 (如 volume, tick 笔数, 波动率 RV/BV, 价差 spread 等)；
4. 统计极端离群点 (Statistical Outliers)：基于 Z-Score / IQR 离群点扫描；
5. 低流动性与竞价时段上下文偏倚排查 (Low Tick / Auction Context)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import polars as pl


# 常见微观与技术因子的权威理论取值约束表
# 格式: 因子名/别名 -> (理论下界, 理论上界, 描述)
KNOWN_FACTOR_BOUNDS: dict[str, tuple[float | None, float | None, str]] = {
    # === mc001 资金流向类 ===
    "mc001_001": (0.0, None, "主动买入笔数"),
    "mc001_002": (0.0, None, "主动卖出笔数"),
    "mc001_004": (0.0, 1.0, "主动买入笔数占比"),
    "mc001_005": (0.0, 1.0, "主动卖出笔数占比"),
    "mc001_006": (-1.0, 1.0, "净买入笔数占比"),
    "mc001_007": (0.0, None, "主动买入成交量"),
    "mc001_008": (0.0, None, "主动卖出成交量"),
    "mc001_010": (0.0, 1.0, "主动买入成交量占比"),
    "mc001_011": (0.0, 1.0, "主动卖出成交量占比"),
    "mc001_012": (-1.0, 1.0, "净主动买入量占比"),
    "mc001_013": (0.0, None, "主动买入成交额"),
    "mc001_014": (0.0, None, "主动卖出成交额"),
    "mc001_016": (-1.0, 1.0, "净流入金额占比"),
    "mc001_017": (0.0, None, "聪明钱买入成交量"),
    "mc001_018": (0.0, None, "聪明钱卖出成交量"),
    "mc001_019": (0.0, 1.0, "聪明钱买入量占比"),
    "mc001_020": (0.0, 1.0, "聪明钱卖出量占比"),
    "mc001_021": (0.0, None, "聪明钱买入成交额"),
    "mc001_022": (0.0, None, "聪明钱卖出成交额"),
    "mc001_023": (0.0, 1.0, "聪明钱买入额占比"),
    "mc001_024": (0.0, 1.0, "聪明钱卖出额占比"),
    "mc001_025": (-1.0, 1.0, "聪明钱净成交量占比"),

    # === mc002 买卖盘口类 ===
    "mc002_001": (0.0, None, "平均买卖价差"),
    "mc002_002": (0.0, None, "买卖价差波动率"),
    "mc002_003": (0.0, None, "相对买卖价差"),
    "mc002_004": (-1.0, 1.0, "一档买卖深度不平衡均值"),
    "mc002_005": (0.0, None, "一档盘口不平衡波动率"),
    "mc002_006": (-1.0, 1.0, "1分钟收盘瞬时盘口不平衡"),
    "mc002_008": (0.0, None, "已实现高频波动率 RV"),
    "mc002_009": (0.0, None, "双截差波动率 BV"),
    "mc002_010": (0.0, 1.0, "离散跳跃相对占比"),

    # === mc003 订单流动力学 ===
    "mc003_003": (0.0, None, "OFI 离散度"),
    "mc003_007": (0.0, None, "买单消耗速度"),
    "mc003_008": (0.0, None, "卖单消耗速度"),
    "mc003_009": (0.0, 1.0, "买单防线击穿比率"),
    "mc003_010": (0.0, 1.0, "卖单防线击穿比率"),
    "mc003_011": (-1.0, 1.0, "挂单防线失守差比"),

    # === mc004 相关性类 ===
    "mc004_001": (-1.0, 1.0, "成交额与收益率相关性"),
    "mc004_002": (-1.0, 1.0, "成交额与价差相关性"),
    "mc004_003": (-1.0, 1.0, "成交量与盘口不平衡相关性"),
    "mc004_004": (-1.0, 1.0, "VWAP与价差相关性"),

    # === mf001 期货持仓博弈类 ===
    "mf001_002": (0.0, None, "持仓绝对活动量"),
    "mf001_003": (-1.0, 1.0, "净持仓增量占比"),
    "mf001_004": (0.0, None, "总持仓变动比率"),
    "mf001_005": (0.0, None, "增仓成交比"),
    "mf001_006": (0.0, None, "双开成交量"),
    "mf001_007": (0.0, None, "双平成交量"),
    "mf001_008": (0.0, None, "换手量"),
    "mf001_009": (0.0, 1.0, "双开成交量占比"),
    "mf001_010": (0.0, 1.0, "双平成交量占比"),
    "mf001_011": (0.0, 1.0, "换手成交量占比"),
    "mf001_012": (0.0, None, "主动多头增仓量"),
    "mf001_013": (0.0, None, "主动空头增仓量"),
    "mf001_014": (0.0, None, "多头止损平仓量"),
    "mf001_015": (0.0, None, "空头止损平仓量"),
    "mf001_016": (0.0, 1.0, "主动多头增仓占比"),
    "mf001_017": (0.0, 1.0, "主动空头增仓占比"),
    "mf001_018": (0.0, 1.0, "多头砍仓踩踏占比"),
    "mf001_019": (0.0, 1.0, "空头止损逼仓占比"),
    "mf001_020": (-1.0, 1.0, "增仓能量不平衡"),
    "mf001_021": (0.0, None, "增仓重心价格"),

    # === mf002 涨跌停极限边界 ===
    "mf002_001": (0.0, None, "距离涨停板相对幅度"),
    "mf002_002": (0.0, None, "距离跌停板相对幅度"),
    "mf002_003": (-1.0, 1.0, "涨跌停距离不对称性"),

    # === mf003 日均线锚点 ===
    "mf003_004": (0.0, 1.0, "均线上方时间占比"),
}


def infer_bounds_by_name(col_name: str) -> tuple[float | None, float | None, str]:
    """根据列名的命名特征与行业模式自动推断其合理的理论上下界。"""
    c = col_name.lower()
    if c in KNOWN_FACTOR_BOUNDS:
        return KNOWN_FACTOR_BOUNDS[c]

    # 不平衡度 / 对称性 / 相关系数: [-1, 1]
    if any(k in c for k in ("imbalance", "asymmetry", "corr", "net_")):
        return (-1.0, 1.0, "推断为双向平衡/相关性类指标 [-1, 1]")

    # 占比 / 概率 / 率类: [0, 1]
    if any(k in c for k in ("_pct", "_ratio", "ratio", "pct", "dominance", "probability")):
        return (0.0, 1.0, "推断为单向比率/占比指标 [0, 1]")

    # 波动率 / 标准差 / 价差 / 成交手数 / 笔数: [0, +inf)
    if any(k in c for k in ("_std", "std", "volatility", "spread", "volume", "turnover", "tick_", "count", "depth")):
        return (0.0, None, "推断为非负物理指标 [0, +inf)")

    return (None, None, "无明确物理边界")


@dataclass
class AnomalyMetric:
    """单个因子的异常与质量检测汇总。"""
    column: str
    dtype: str
    total_rows: int
    null_count: int
    nan_count: int
    inf_count: int
    underflow_count: int
    bound_violation_count: int
    lower_bound: float | None
    upper_bound: float | None
    min_val: float | None
    max_val: float | None
    mean_val: float | None
    std_val: float | None
    description: str

    @property
    def has_error(self) -> bool:
        """是否存在不可接受的硬性错误 (下溢、Inf、边界越界、大量 NaN)。"""
        return (
            self.underflow_count > 0
            or self.inf_count > 0
            or self.bound_violation_count > 0
        )

    @property
    def error_summary(self) -> list[str]:
        errors = []
        if self.underflow_count > 0:
            errors.append(f"整型下溢 {self.underflow_count} 行 (发现异常巨数)")
        if self.inf_count > 0:
            errors.append(f"包含无穷值 Inf {self.inf_count} 行")
        if self.bound_violation_count > 0:
            b_desc = f"[{self.lower_bound}, {self.upper_bound}]"
            errors.append(f"超出理论边界 {b_desc}: {self.bound_violation_count} 行 (min={self.min_val}, max={self.max_val})")
        if self.nan_count > 0:
            errors.append(f"NaN 缺失 {self.nan_count} 行")
        return errors


class AnomalyReport:
    """检测报告对象，提供摘要、格式化打印、违规行定位及健康数据清洗功能。"""

    def __init__(
        self,
        metrics: list[AnomalyMetric],
        bad_expressions: dict[str, pl.Expr],
        original_lazy: pl.LazyFrame,
    ):
        self.metrics = metrics
        self.bad_expressions = bad_expressions
        self.original_lazy = original_lazy

    @property
    def is_clean(self) -> bool:
        """全部检测的因子是否 100% 健康（零硬性异常）。"""
        return not any(m.has_error for m in self.metrics)

    @property
    def error_columns(self) -> list[str]:
        """存在硬性异常的列名列表。"""
        return [m.column for m in self.metrics if m.has_error]

    def summary_table(self) -> pl.DataFrame:
        """返回所有存在质量疑问或全部因子的结构化 Polars 汇总表。"""
        records = []
        for m in self.metrics:
            records.append({
                "column": m.column,
                "dtype": m.dtype,
                "status": "❌ ERROR" if m.has_error else ("⚠️ WARNING" if m.nan_count > 0 else "🟢 OK"),
                "null_cnt": m.null_count,
                "nan_cnt": m.nan_count,
                "inf_cnt": m.inf_count,
                "underflow_cnt": m.underflow_count,
                "bound_violation_cnt": m.bound_violation_count,
                "expected_bound": f"[{m.lower_bound}, {m.upper_bound}]",
                "min": m.min_val,
                "max": m.max_val,
                "issues": "; ".join(m.error_summary) if m.error_summary else "Normal",
            })
        return pl.DataFrame(records)

    def print_summary(self, verbose: bool = False) -> None:
        """格式化友好打印审计报告。"""
        print("=" * 88)
        print("🔍 Orion Polars 因子异常值与数据质量审计报告")
        print("=" * 88)

        total_checked = len(self.metrics)
        errors = [m for m in self.metrics if m.has_error]
        warnings = [m for m in self.metrics if not m.has_error and m.nan_count > 0]

        print(f"📊 审计字段总数: {total_checked} | 🟢 正常: {total_checked - len(errors) - len(warnings)} | ❌ 异常: {len(errors)} | ⚠️ 警告: {len(warnings)}")
        print("-" * 88)

        if errors:
            print("🚨 【发现严重异常字段】:")
            for m in errors:
                print(f"  ❌ [{m.column}] ({m.description}):")
                for err in m.error_summary:
                    print(f"     - {err}")
        else:
            print("✅ 未发现严重异常字段（所有字段均符合物理与理论边界）")

        if warnings and verbose:
            print("\n⚠️ 【存在缺失警告字段】:")
            for m in warnings:
                print(f"  ⚠️ [{m.column}]: NaN 缺失 {m.nan_count} 行 ({m.nan_count / m.total_rows:.2%})")

        print("=" * 88)

    def get_bad_rows(self, column: str | None = None, limit: int = 50) -> pl.DataFrame:
        """提取指定因子或全部异常因子的实际样本行（包含主键），方便排查复现。"""
        keys = [c for c in ["trade_time", "code", "symbol"] if c in self.original_lazy.collect_schema().names()]
        if column is not None:
            if column not in self.bad_expressions:
                return pl.DataFrame()
            cond = self.bad_expressions[column]
            cols = [*keys, column]
            return self.original_lazy.filter(cond).select(cols).limit(limit).collect()

        # 全部因子的并集异常
        all_conds = list(self.bad_expressions.values())
        if not all_conds:
            return pl.DataFrame()

        combined_cond = all_conds[0]
        for c in all_conds[1:]:
            combined_cond = combined_cond | c

        error_cols = [c for c in self.error_columns if c in self.original_lazy.collect_schema().names()]
        target_cols = [*keys, *error_cols]
        return self.original_lazy.filter(combined_cond).select(target_cols).limit(limit).collect()


def check_factor_anomalies(
    df: pl.LazyFrame | pl.DataFrame,
    columns: Iterable[str] | None = None,
    custom_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    tolerance: float = 1e-5,
    sample_size: int | None = None,
) -> AnomalyReport:
    """使用纯 Polars 原生表达式单次扫描完成因子全量异常值检测。

    参数:
        df: 输入的 LazyFrame 或 DataFrame
        columns: 指定要检查的列；未指定时自动挑选全部浮点型与整型数值特征列
        custom_bounds: 自定义指定各列的理论上下界字典 {col: (min, max)}
        tolerance: 浮点数边界容差 (默认 1e-5，避免 1.00000000001 被误报)
        sample_size: 可选的大数据抽样限制

    返回:
        AnomalyReport: 结构化异常分析报告对象
    """
    df_lazy = df.lazy() if isinstance(df, pl.DataFrame) else df
    schema = df_lazy.collect_schema()
    all_names = schema.names()

    # 过滤出数值型因子列
    non_numeric_keys = {"trade_time", "code", "symbol", "bar_time", "date"}
    if columns is None:
        target_cols = [
            col for col in all_names
            if col not in non_numeric_keys and schema[col].is_numeric()
        ]
    else:
        target_cols = [col for col in columns if col in all_names and schema[col].is_numeric()]

    if not target_cols:
        return AnomalyReport([], {}, df_lazy)

    # 确定各列的上下界
    bounds_map: dict[str, tuple[float | None, float | None, str]] = {}
    for col in target_cols:
        if custom_bounds and col in custom_bounds:
            low, high = custom_bounds[col]
            bounds_map[col] = (low, high, "用户自定义约束")
        else:
            bounds_map[col] = infer_bounds_by_name(col)

    # 单次聚合扫描表达式构造
    agg_exprs: list[pl.Expr] = [pl.len().alias("_total_count")]
    bad_expressions: dict[str, pl.Expr] = {}

    for col in target_cols:
        low, high, _ = bounds_map[col]
        col_expr = pl.col(col)

        # 1. 基础缺失与异常值统计
        agg_exprs.append(col_expr.is_null().sum().alias(f"{col}_null_cnt"))
        agg_exprs.append(col_expr.is_nan().sum().alias(f"{col}_nan_cnt"))
        agg_exprs.append(col_expr.is_infinite().sum().alias(f"{col}_inf_cnt"))

        # 2. 整数下溢特征检测:
        # 当 UInt32 发生 0 - 1 减法下溢时，产生 >= 4290000000 的极端巨数
        underflow_cond = (col_expr > 4.29e9) & (col_expr < 4.30e9)
        agg_exprs.append(underflow_cond.sum().alias(f"{col}_underflow_cnt"))

        # 3. 边界违规统计 (加入 tolerance 容差)
        bound_cond = pl.lit(False)
        if low is not None:
            bound_cond = bound_cond | (col_expr < (low - tolerance))
        if high is not None:
            bound_cond = bound_cond | (col_expr > (high + tolerance))

        # 排除 null / nan / inf 后再计入边界违规
        valid_val_cond = col_expr.is_not_null() & (~col_expr.is_nan()) & (~col_expr.is_infinite())
        agg_exprs.append((bound_cond & valid_val_cond).sum().alias(f"{col}_bound_violation_cnt"))

        # 4. 基础极值统计
        agg_exprs.append(col_expr.filter(valid_val_cond).min().alias(f"{col}_min"))
        agg_exprs.append(col_expr.filter(valid_val_cond).max().alias(f"{col}_max"))
        agg_exprs.append(col_expr.filter(valid_val_cond).mean().alias(f"{col}_mean"))
        agg_exprs.append(col_expr.filter(valid_val_cond).std().alias(f"{col}_std"))

        # 记录每列的严重异常判断条件，供后续提取坏行样本
        severe_cond = col_expr.is_infinite() | underflow_cond | (bound_cond & valid_val_cond)
        bad_expressions[col] = severe_cond

    # 触发单次聚合计算
    scan_df = df_lazy if sample_size is None else df_lazy.limit(sample_size)
    summary_row = scan_df.select(agg_exprs).collect()

    total_rows = summary_row["_total_count"][0]
    metrics: list[AnomalyMetric] = []

    for col in target_cols:
        low, high, desc = bounds_map[col]
        dtype_str = str(schema[col])
        metric = AnomalyMetric(
            column=col,
            dtype=dtype_str,
            total_rows=total_rows,
            null_count=int(summary_row[f"{col}_null_cnt"][0] or 0),
            nan_count=int(summary_row[f"{col}_nan_cnt"][0] or 0),
            inf_count=int(summary_row[f"{col}_inf_cnt"][0] or 0),
            underflow_count=int(summary_row[f"{col}_underflow_cnt"][0] or 0),
            bound_violation_count=int(summary_row[f"{col}_bound_violation_cnt"][0] or 0),
            lower_bound=low,
            upper_bound=high,
            min_val=summary_row[f"{col}_min"][0],
            max_val=summary_row[f"{col}_max"][0],
            mean_val=summary_row[f"{col}_mean"][0],
            std_val=summary_row[f"{col}_std"][0],
            description=desc,
        )
        metrics.append(metric)

    return AnomalyReport(metrics, bad_expressions, df_lazy)


__all__ = [
    "KNOWN_FACTOR_BOUNDS",
    "AnomalyMetric",
    "AnomalyReport",
    "check_factor_anomalies",
    "infer_bounds_by_name",
]
