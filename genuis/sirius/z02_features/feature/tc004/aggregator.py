"""tc004 通用技术成交量分布与换手类因子批次聚合引擎 (Aggregator Engine)。

负责统一组织 tc004_001 ~ tc004_021 全部 21 个技术因子，
提供纯 Polars Lazy 惰性计算图与批量对齐入口。
输出标准主键: ['trade_time', 'code'] 及各因子列。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import polars as pl

from feature.utils.common import KEY_COLUMNS, FactorCompute, compute_factor_batch

from .tc004_001 import compute as tc004_001_compute
from .tc004_002 import compute as tc004_002_compute
from .tc004_003 import compute as tc004_003_compute
from .tc004_004 import compute as tc004_004_compute
from .tc004_005 import compute as tc004_005_compute
from .tc004_006 import compute as tc004_006_compute
from .tc004_007 import compute as tc004_007_compute
from .tc004_008 import compute as tc004_008_compute
from .tc004_009 import compute as tc004_009_compute
from .tc004_010 import compute as tc004_010_compute
from .tc004_011 import compute as tc004_011_compute
from .tc004_012 import compute as tc004_012_compute
from .tc004_013 import compute as tc004_013_compute
from .tc004_014 import compute as tc004_014_compute
from .tc004_015 import compute as tc004_015_compute
from .tc004_016 import compute as tc004_016_compute
from .tc004_017 import compute as tc004_017_compute
from .tc004_018 import compute as tc004_018_compute
from .tc004_019 import compute as tc004_019_compute
from .tc004_020 import compute as tc004_020_compute
from .tc004_021 import compute as tc004_021_compute

FACTOR_NAMES: list[str] = [f"tc004_{i:03d}" for i in range(1, 22)]

FACTORS: dict[str, FactorCompute] = {
    "tc004_001": tc004_001_compute,
    "tc004_002": tc004_002_compute,
    "tc004_003": tc004_003_compute,
    "tc004_004": tc004_004_compute,
    "tc004_005": tc004_005_compute,
    "tc004_006": tc004_006_compute,
    "tc004_007": tc004_007_compute,
    "tc004_008": tc004_008_compute,
    "tc004_009": tc004_009_compute,
    "tc004_010": tc004_010_compute,
    "tc004_011": tc004_011_compute,
    "tc004_012": tc004_012_compute,
    "tc004_013": tc004_013_compute,
    "tc004_014": tc004_014_compute,
    "tc004_015": tc004_015_compute,
    "tc004_016": tc004_016_compute,
    "tc004_017": tc004_017_compute,
    "tc004_018": tc004_018_compute,
    "tc004_019": tc004_019_compute,
    "tc004_020": tc004_020_compute,
    "tc004_021": tc004_021_compute,
}

FACTOR_ALIASES: dict[str, str] = {
    "tc004_001": "dv001",
    "tc004_002": "dv002",
    "tc004_003": "dv003",
    "tc004_004": "dv004",
    "tc004_005": "dv005",
    "tc004_006": "dv006",
    "tc004_007": "dv007",
    "tc004_008": "dv008",
    "tc004_009": "dv009",
    "tc004_010": "dv010",
    "tc004_011": "dv011",
    "tc004_012": "dv012",
    "tc004_013": "tn001",
    "tc004_014": "tn002",
    "tc004_015": "tn003",
    "tc004_016": "tn004",
    "tc004_017": "tn005",
    "tc004_018": "tn006",
    "tc004_019": "tn007",
    "tc004_020": "tn008",
    "tc004_021": "tn009",
}

def aggregate_tc004(
    df_lazy: pl.LazyFrame,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
    use_aliases: bool = False,
) -> pl.LazyFrame:
    """批量计算 tc004 批次因子；未指定 names 时计算全部 21 个因子。"""
    selected = tuple(FACTOR_NAMES) if names is None else tuple(names)
    res = compute_factor_batch(
        df_lazy,
        FACTORS,
        names=selected,
        params=params,
        key_columns=KEY_COLUMNS,
    )
    if use_aliases:
        schema_names = res.collect_schema().names()
        rename_map = {k: v for k, v in FACTOR_ALIASES.items() if k in schema_names}
        return res.rename(rename_map)
    return res


compute = aggregate_tc004

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "FACTORS",
    "aggregate_tc004",
    "compute",
]
