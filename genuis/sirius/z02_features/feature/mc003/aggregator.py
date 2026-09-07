"""
mc003 一档订单流动力学类特征下采样聚合引擎 (Downsampling Aggregator Engine)。

负责统一组织 mc003_001 ~ mc003_011 因子文件中定义的核心计算表达式，
单次扫描 Tick 数据，一次性高效聚合全部 11 个一档订单流动力学特征。
输出标准化维度列: trade_time, code (如 RB), symbol (如 rb1205)。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mc003_001 import EXPR as mc003_001_expr
from .mc003_002 import EXPR as mc003_002_expr
from .mc003_003 import EXPR as mc003_003_expr
from .mc003_004 import EXPR as mc003_004_expr
from .mc003_005 import EXPR as mc003_005_expr
from .mc003_006 import EXPR as mc003_006_expr
from .mc003_007 import EXPR as mc003_007_expr
from .mc003_008 import EXPR as mc003_008_expr
from .mc003_009 import EXPR as mc003_009_expr
from .mc003_010 import EXPR as mc003_010_expr
from .mc003_011 import EXPR as mc003_011_expr

FACTOR_NAMES = [f"mc003_{i:03d}" for i in range(1, 12)]

FACTOR_ALIASES: dict[str, str] = {
    "mc003_001": "ofi_sum",
    "mc003_002": "ofi_mean",
    "mc003_003": "ofi_std",
    "mc003_004": "ofi_normalized",
    "mc003_005": "voi_1",
    "mc003_006": "voi_normalized",
    "mc003_007": "bid_depletion_rate",
    "mc003_008": "ask_depletion_rate",
    "mc003_009": "bid_depletion_ratio",
    "mc003_010": "ask_depletion_ratio",
    "mc003_011": "depletion_imbalance",
}


def aggregate_mc003(df_lazy: pl.LazyFrame, use_aliases: bool = False) -> pl.LazyFrame:
    """全量一档订单流动力学特征聚合计算引擎。"""
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
            mc003_001_expr,
            mc003_002_expr,
            mc003_003_expr,
            mc003_004_expr,
            mc003_005_expr,
            mc003_006_expr,
            mc003_007_expr,
            mc003_008_expr,
            mc003_009_expr,
            mc003_010_expr,
            mc003_011_expr,
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


compute = aggregate_mc003

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "aggregate_mc003",
    "compute",
]
