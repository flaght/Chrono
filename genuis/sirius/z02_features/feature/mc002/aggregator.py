"""
mc002 买卖盘口与微观结构类特征下采样聚合引擎 (Downsampling Aggregator Engine)。

负责统一组织 mc002_001 ~ mc002_010 因子文件中定义的核心计算表达式，
单次扫描 Tick 数据，一次性高效聚合全部 10 个买卖盘口与微观结构类特征。
输出标准化维度列: trade_time, code (如 RB), symbol (如 rb1205)。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mc002_001 import EXPR as mc002_001_expr
from .mc002_002 import EXPR as mc002_002_expr
from .mc002_003 import EXPR as mc002_003_expr
from .mc002_004 import EXPR as mc002_004_expr
from .mc002_005 import EXPR as mc002_005_expr
from .mc002_006 import EXPR as mc002_006_expr
from .mc002_007 import EXPR as mc002_007_expr
from .mc002_008 import EXPR as mc002_008_expr
from .mc002_009 import EXPR as mc002_009_expr
from .mc002_010 import EXPR as mc002_010_expr

FACTOR_NAMES = [f"mc002_{i:03d}" for i in range(1, 11)]

FACTOR_ALIASES: dict[str, str] = {
    "mc002_001": "bid_ask_spread",
    "mc002_002": "spread_std",
    "mc002_003": "relative_spread",
    "mc002_004": "depth_imbalance_1",
    "mc002_005": "depth_imbalance_std",
    "mc002_006": "depth_imbalance_last",
    "mc002_007": "micro_price_bias",
    "mc002_008": "realized_volatility",
    "mc002_009": "realized_bipower_var",
    "mc002_010": "jump_ratio",
}


def aggregate_mc002(df_lazy: pl.LazyFrame, use_aliases: bool = False) -> pl.LazyFrame:
    """全量微观结构与盘口特征聚合计算引擎。"""
    schema_names = df_lazy.collect_schema().names()
    if "_spread" not in schema_names or "symbol" not in schema_names:
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
            mc002_001_expr,
            mc002_002_expr,
            mc002_003_expr,
            mc002_004_expr,
            mc002_005_expr,
            mc002_006_expr,
            mc002_007_expr,
            mc002_008_expr,
            mc002_009_expr,
        ])
        .with_columns([
            mc002_010_expr,
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


compute = aggregate_mc002

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "aggregate_mc002",
    "compute",
]
