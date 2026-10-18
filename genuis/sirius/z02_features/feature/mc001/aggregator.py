"""
mc001 资金流向高频特征下采样聚合引擎 (Downsampling Aggregator Engine)。

负责统一组织 mc001_001 ~ mc001_025 因子文件中定义的核心计算表达式，
单次扫描 Tick 数据，一次性高效聚合全部资金流特征。
输出标准化维度列: trade_time, code (如 RB), symbol (如 rb1205)。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 直接从各单因子文件中导入其核心计算表达式（各因子模块自持计算公式）
from .mc001_001 import EXPR as mc001_001_expr
from .mc001_002 import EXPR as mc001_002_expr
from .mc001_003 import EXPR as mc001_003_expr
from .mc001_004 import EXPR as mc001_004_expr
from .mc001_005 import EXPR as mc001_005_expr
from .mc001_006 import EXPR as mc001_006_expr
from .mc001_007 import EXPR as mc001_007_expr
from .mc001_008 import EXPR as mc001_008_expr
from .mc001_009 import EXPR as mc001_009_expr
from .mc001_010 import EXPR as mc001_010_expr
from .mc001_011 import EXPR as mc001_011_expr
from .mc001_012 import EXPR as mc001_012_expr
from .mc001_013 import EXPR as mc001_013_expr
from .mc001_014 import EXPR as mc001_014_expr
from .mc001_015 import EXPR as mc001_015_expr
from .mc001_016 import EXPR as mc001_016_expr
from .mc001_017 import EXPR as mc001_017_expr
from .mc001_018 import EXPR as mc001_018_expr
from .mc001_019 import EXPR as mc001_019_expr
from .mc001_020 import EXPR as mc001_020_expr
from .mc001_021 import EXPR as mc001_021_expr
from .mc001_022 import EXPR as mc001_022_expr
from .mc001_023 import EXPR as mc001_023_expr
from .mc001_024 import EXPR as mc001_024_expr
from .mc001_025 import EXPR as mc001_025_expr

FACTOR_NAMES = [f"mc001_{i:03d}" for i in range(1, 26)]

FACTOR_ALIASES: dict[str, str] = {
    "mc001_001": "tick_in",
    "mc001_002": "tick_out",
    "mc001_003": "net_tick_in",
    "mc001_004": "tick_in_pct",
    "mc001_005": "tick_out_pct",
    "mc001_006": "net_tick_in_pct",
    "mc001_007": "volume_in",
    "mc001_008": "volume_out",
    "mc001_009": "net_volume_in",
    "mc001_010": "volume_in_pct",
    "mc001_011": "volume_out_pct",
    "mc001_012": "net_volume_in_pct",
    "mc001_013": "money_in",
    "mc001_014": "money_out",
    "mc001_015": "net_money_in",
    "mc001_016": "net_money_in_pct",
    "mc001_017": "smart_volume_in",
    "mc001_018": "smart_volume_out",
    "mc001_019": "smart_volume_in_pct",
    "mc001_020": "smart_volume_out_pct",
    "mc001_021": "smart_money_in",
    "mc001_022": "smart_money_out",
    "mc001_023": "smart_money_in_pct",
    "mc001_024": "smart_money_out_pct",
    "mc001_025": "smart_net_vol_pct",
}


def aggregate_mc001(df_lazy: pl.LazyFrame, use_aliases: bool = False) -> pl.LazyFrame:
    """全量微观资金流特征聚合计算引擎：

    单次扫描 Tick 数据，一次性高并行聚合出全部 25 个资金流向特征。
    所有特征表达式直接引用自各因子文件。
    输出包含 trade_time, code (品种代码), symbol (合约代码) 以及特征列。
    """
    # 步骤 1: 检查是否已包含基础预处理衍生列，若未包含则自动调用 preprocess_ticks
    schema_names = df_lazy.collect_schema().names()
    if "_trade_dir" not in schema_names or "symbol" not in schema_names:
        primitives = preprocess_ticks(df_lazy)
    else:
        primitives = df_lazy

    res = (
        primitives
        .with_columns([
            # 步骤 2: 时间戳对齐至 1 分钟开始时刻 (trade_time)
            pl.col("timestamp").dt.truncate("1m").alias("trade_time"),
        ])
        .group_by(["trade_time", "code", "symbol"])
        .agg([
            # 步骤 3: 基础高频统计汇总
            pl.len().alias("tick_count"),
            pl.col("_delta_v").sum().alias("volume"),
            pl.col("_delta_m").sum().alias("money"),

            # 步骤 4: 基础买卖笔数、成交量与成交额聚合（各单因子 EXPR 自持定义）
            mc001_001_expr,  # tick_in: 主动买入成交笔数
            mc001_002_expr,  # tick_out: 主动卖出成交笔数
            mc001_003_expr,  # net_tick_in: 净主动买入笔数
            mc001_007_expr,  # volume_in: 主动买入成交量
            mc001_008_expr,  # volume_out: 主动卖出成交量
            mc001_009_expr,  # net_volume_in: 净主动买入成交量
            mc001_013_expr,  # money_in: 主动买入成交额
            mc001_014_expr,  # money_out: 主动卖出成交额
            mc001_015_expr,  # net_money_in: 净主动流入金额
            mc001_017_expr,  # smart_volume_in: 聪明钱买入成交量 (大单 quantile>=0.9)
            mc001_018_expr,  # smart_volume_out: 聪明钱卖出成交量 (大单 quantile>=0.9)
            mc001_021_expr,  # smart_money_in: 聪明钱买入成交额 (大单 quantile>=0.9)
            mc001_022_expr,  # smart_money_out: 聪明钱卖出成交额 (大单 quantile>=0.9)
        ])
        .with_columns([
            # 步骤 5: 二次比率化衍生指标计算（归一化消除量纲干扰，具备天然跨主力可比性）
            mc001_004_expr,  # tick_in_pct: 主动买入笔数占比
            mc001_005_expr,  # tick_out_pct: 主动卖出笔数占比
            mc001_006_expr,  # net_tick_in_pct: 净买入笔数占比
            mc001_010_expr,  # volume_in_pct: 主动买入成交量占比
            mc001_011_expr,  # volume_out_pct: 主动卖出成交量占比
            mc001_012_expr,  # net_volume_in_pct: 净主动买入量占比
            mc001_016_expr,  # net_money_in_pct: 净流入金额占比
            mc001_019_expr,  # smart_volume_in_pct: 聪明钱买入量占比
            mc001_020_expr,  # smart_volume_out_pct: 聪明钱卖出量占比
            mc001_023_expr,  # smart_money_in_pct: 聪明钱买入额占比
            mc001_024_expr,  # smart_money_out_pct: 聪明钱卖出额占比
            mc001_025_expr,  # smart_net_vol_pct: 聪明钱净成交量占比
        ])
        .select([
            # 步骤 6: 选定维度主键与全部 25 个资金流特征列
            "trade_time", "code", "symbol",
            *FACTOR_NAMES,
        ])
        .sort(["trade_time", "code", "symbol"])
    )

    if use_aliases:
        rename_map = {k: v for k, v in FACTOR_ALIASES.items()}
        return res.rename(rename_map)

    return res


compute = aggregate_mc001

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "aggregate_mc001",
    "compute",
]
