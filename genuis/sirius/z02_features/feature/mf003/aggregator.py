"""
mf003 期货日内均价线与基准锚点类特征下采样聚合引擎 (Downsampling Aggregator Engine)。

负责统一组织 mf003_001 ~ mf003_004 因子文件中定义的核心计算表达式，
单次扫描 Tick 数据，一次性高效聚合全部 4 个期货日内均价线与基准锚点特征。
输出标准化维度列: trade_time, code (如 RB), symbol (如 rb1205)。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mf003_001 import EXPR as mf003_001_expr
from .mf003_002 import EXPR as mf003_002_expr
from .mf003_003 import EXPR as mf003_003_expr
from .mf003_004 import EXPR as mf003_004_expr

FACTOR_NAMES = [f"mf003_{i:03d}" for i in range(1, 5)]

FACTOR_ALIASES: dict[str, str] = {
    "mf003_001": "price_to_avg_dev",
    "mf003_002": "vwap_to_avg_bias",
    "mf003_003": "avg_price_slope",
    "mf003_004": "price_above_avg_time",
}


def aggregate_mf003(df_lazy: pl.LazyFrame, use_aliases: bool = False) -> pl.LazyFrame:
    """全量期货日内均价线与基准锚点类特征聚合计算引擎。"""
    schema_names = df_lazy.collect_schema().names()
    if "_above_avg" not in schema_names or "symbol" not in schema_names:
        primitives = preprocess_ticks(df_lazy)
    else:
        primitives = df_lazy

    res = (
        primitives
        .with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("trade_time"),
        ])
        .group_by(["trade_time", "code", "symbol"])
        .agg([
            mf003_001_expr,
            mf003_002_expr,
            mf003_003_expr,
            mf003_004_expr,
        ])
        .select([
            "trade_time", "code", "symbol",
            *FACTOR_NAMES,
        ])
        .sort(["trade_time", "code", "symbol"])
    )

    if use_aliases:
        rename_map = {k: v for k, v in FACTOR_ALIASES.items()}
        return res.rename(rename_map)

    return res


def compute(df_lazy: pl.LazyFrame, use_aliases: bool = False) -> pl.LazyFrame:
    """批量计算全量特征。"""
    return aggregate_mf003(df_lazy, use_aliases=use_aliases)


__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "aggregate_mf003",
    "compute",
]
