"""
mc004 微观交互与协方差类特征下采样聚合引擎 (Downsampling Aggregator Engine)。

负责统一组织 mc004_001 ~ mc004_005 因子文件中定义的核心计算表达式，
单次扫描 Tick 数据，一次性高效聚合全部 5 个微观交互与协方差类特征。
输出标准化维度列: trade_time, code (如 RB), symbol (如 rb1205)。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mc004_001 import EXPR as mc004_001_expr
from .mc004_002 import EXPR as mc004_002_expr
from .mc004_003 import EXPR as mc004_003_expr
from .mc004_004 import EXPR as mc004_004_expr
from .mc004_005 import EXPR as mc004_005_expr

FACTOR_NAMES = [f"mc004_{i:03d}" for i in range(1, 6)]

FACTOR_ALIASES: dict[str, str] = {
    "mc004_001": "corr_money_ret",
    "mc004_002": "corr_money_spread",
    "mc004_003": "corr_ofi_ret",
    "mc004_004": "corr_vwap_spread",
    "mc004_005": "corr_vol_depth_imb",
}


def aggregate_mc004(df_lazy: pl.LazyFrame, use_aliases: bool = False) -> pl.LazyFrame:
    """全量微观交互与协方差类特征聚合计算引擎。"""
    schema_names = df_lazy.collect_schema().names()
    if "_ofi" not in schema_names or "symbol" not in schema_names:
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
            mc004_001_expr,
            mc004_002_expr,
            mc004_003_expr,
            mc004_004_expr,
            mc004_005_expr,
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


compute = aggregate_mc004

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "aggregate_mc004",
    "compute",
]

