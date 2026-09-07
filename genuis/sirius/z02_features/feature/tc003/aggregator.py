"""tc003 通用技术波动率与通道带类因子批次聚合引擎 (Aggregator Engine)。

负责统一组织 tc003_001 ~ tc003_015 全部 15 个技术因子，
提供纯 Polars Lazy 惰性计算图与批量对齐入口。
输出标准主键: ['trade_time', 'code'] 及各因子列。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import polars as pl

from feature.utils.common import KEY_COLUMNS, FactorCompute, compute_factor_batch

from .tc003_001 import compute as tc003_001_compute
from .tc003_002 import compute as tc003_002_compute
from .tc003_003 import compute as tc003_003_compute
from .tc003_004 import compute as tc003_004_compute
from .tc003_005 import compute as tc003_005_compute
from .tc003_006 import compute as tc003_006_compute
from .tc003_007 import compute as tc003_007_compute
from .tc003_008 import compute as tc003_008_compute
from .tc003_009 import compute as tc003_009_compute
from .tc003_010 import compute as tc003_010_compute
from .tc003_011 import compute as tc003_011_compute
from .tc003_012 import compute as tc003_012_compute
from .tc003_013 import compute as tc003_013_compute
from .tc003_014 import compute as tc003_014_compute
from .tc003_015 import compute as tc003_015_compute

FACTOR_NAMES: list[str] = [f"tc003_{i:03d}" for i in range(1, 16)]

FACTORS: dict[str, FactorCompute] = {
    "tc003_001": tc003_001_compute,
    "tc003_002": tc003_002_compute,
    "tc003_003": tc003_003_compute,
    "tc003_004": tc003_004_compute,
    "tc003_005": tc003_005_compute,
    "tc003_006": tc003_006_compute,
    "tc003_007": tc003_007_compute,
    "tc003_008": tc003_008_compute,
    "tc003_009": tc003_009_compute,
    "tc003_010": tc003_010_compute,
    "tc003_011": tc003_011_compute,
    "tc003_012": tc003_012_compute,
    "tc003_013": tc003_013_compute,
    "tc003_014": tc003_014_compute,
    "tc003_015": tc003_015_compute,
}

FACTOR_ALIASES: dict[str, str] = {
    "tc003_001": "fz002",
    "tc003_002": "gd002",
    "tc003_003": "gd003",
    "tc003_004": "ha004",
    "tc003_005": "ha005",
    "tc003_006": "tf001",
    "tc003_007": "tf002",
    "tc003_008": "tf003",
    "tc003_009": "tf004",
    "tc003_010": "tf005",
    "tc003_011": "tf006",
    "tc003_012": "tf008",
    "tc003_013": "tf019",
    "tc003_014": "tf020",
    "tc003_015": "tf022",
}

def aggregate_tc003(
    df_lazy: pl.LazyFrame,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
    use_aliases: bool = False,
) -> pl.LazyFrame:
    """批量计算 tc003 批次因子；未指定 names 时计算全部 15 个因子。"""
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


compute = aggregate_tc003

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "FACTORS",
    "aggregate_tc003",
    "compute",
]
