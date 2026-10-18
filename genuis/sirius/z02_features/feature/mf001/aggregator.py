"""
mf001 期货持仓与博弈形态类特征下采样聚合引擎 (Downsampling Aggregator Engine)。

负责统一组织 mf001_001 ~ mf001_021 因子文件中定义的核心计算表达式，
单次扫描 Tick 数据，一次性高效聚合全部 21 个期货持仓与博弈形态特征。
输出标准化维度列: trade_time, code (如 RB), symbol (如 rb1205)。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mf001_001 import EXPR as mf001_001_expr
from .mf001_002 import EXPR as mf001_002_expr
from .mf001_003 import EXPR as mf001_003_expr
from .mf001_004 import EXPR as mf001_004_expr
from .mf001_005 import EXPR as mf001_005_expr
from .mf001_006 import EXPR as mf001_006_expr
from .mf001_007 import EXPR as mf001_007_expr
from .mf001_008 import EXPR as mf001_008_expr
from .mf001_009 import EXPR as mf001_009_expr
from .mf001_010 import EXPR as mf001_010_expr
from .mf001_011 import EXPR as mf001_011_expr
from .mf001_012 import EXPR as mf001_012_expr
from .mf001_013 import EXPR as mf001_013_expr
from .mf001_014 import EXPR as mf001_014_expr
from .mf001_015 import EXPR as mf001_015_expr
from .mf001_016 import EXPR as mf001_016_expr
from .mf001_017 import EXPR as mf001_017_expr
from .mf001_018 import EXPR as mf001_018_expr
from .mf001_019 import EXPR as mf001_019_expr
from .mf001_020 import EXPR as mf001_020_expr
from .mf001_021 import EXPR as mf001_021_expr

FACTOR_NAMES = [f"mf001_{i:03d}" for i in range(1, 22)]

FACTOR_ALIASES: dict[str, str] = {
    "mf001_001": "delta_oi",
    "mf001_002": "abs_delta_oi",
    "mf001_003": "delta_oi_ratio",
    "mf001_004": "abs_delta_oi_ratio",
    "mf001_005": "oi_volume_ratio",
    "mf001_006": "double_open_vol",
    "mf001_007": "double_close_vol",
    "mf001_008": "swap_volume",
    "mf001_009": "double_open_ratio",
    "mf001_010": "double_close_ratio",
    "mf001_011": "swap_volume_ratio",
    "mf001_012": "bull_active_open",
    "mf001_013": "bear_active_open",
    "mf001_014": "bull_stop_loss_vol",
    "mf001_015": "bear_stop_loss_vol",
    "mf001_016": "bull_active_open_ratio",
    "mf001_017": "bear_active_open_ratio",
    "mf001_018": "bull_stop_loss_ratio",
    "mf001_019": "bear_stop_loss_ratio",
    "mf001_020": "oi_flow_imbalance",
    "mf001_021": "oi_weighted_price",
}


def aggregate_mf001(df_lazy: pl.LazyFrame, use_aliases: bool = False) -> pl.LazyFrame:
    """全量期货持仓与博弈形态类特征聚合计算引擎。"""
    schema_names = df_lazy.collect_schema().names()
    if "_delta_oi" not in schema_names or "symbol" not in schema_names:
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
            mf001_001_expr,
            mf001_002_expr,
            mf001_003_expr,
            mf001_004_expr,
            mf001_005_expr,
            mf001_006_expr,
            mf001_007_expr,
            mf001_008_expr,
            mf001_009_expr,
            mf001_010_expr,
            mf001_011_expr,
            mf001_012_expr,
            mf001_013_expr,
            mf001_014_expr,
            mf001_015_expr,
            mf001_016_expr,
            mf001_017_expr,
            mf001_018_expr,
            mf001_019_expr,
            mf001_020_expr,
            mf001_021_expr,
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
    return aggregate_mf001(df_lazy, use_aliases=use_aliases)


__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "aggregate_mf001",
    "compute",
]
