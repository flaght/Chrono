"""tf002 期货期限结构与量仓共振因子批次聚合引擎 (Aggregator Engine)。

负责统一组织 tf002_001 ~ tf002_015 全部 15 个期货特征因子，
提供纯 Polars Lazy 惰性计算图与批量对齐入口。
输出标准主键: ['trade_time', 'code'] 及各因子列。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import polars as pl

from feature.utils.common import KEY_COLUMNS, FactorCompute, compute_factor_batch

from .tf002_001 import compute as tf002_001_compute
from .tf002_002 import compute as tf002_002_compute
from .tf002_003 import compute as tf002_003_compute
from .tf002_004 import compute as tf002_004_compute
from .tf002_005 import compute as tf002_005_compute
from .tf002_006 import compute as tf002_006_compute
from .tf002_007 import compute as tf002_007_compute
from .tf002_008 import compute as tf002_008_compute
from .tf002_009 import compute as tf002_009_compute
from .tf002_010 import compute as tf002_010_compute
from .tf002_011 import compute as tf002_011_compute
from .tf002_012 import compute as tf002_012_compute
from .tf002_013 import compute as tf002_013_compute
from .tf002_014 import compute as tf002_014_compute
from .tf002_015 import compute as tf002_015_compute

FACTOR_NAMES: list[str] = [f"tf002_{i:03d}" for i in range(1, 16)]

FACTORS: dict[str, FactorCompute] = {
    "tf002_001": tf002_001_compute,
    "tf002_002": tf002_002_compute,
    "tf002_003": tf002_003_compute,
    "tf002_004": tf002_004_compute,
    "tf002_005": tf002_005_compute,
    "tf002_006": tf002_006_compute,
    "tf002_007": tf002_007_compute,
    "tf002_008": tf002_008_compute,
    "tf002_009": tf002_009_compute,
    "tf002_010": tf002_010_compute,
    "tf002_011": tf002_011_compute,
    "tf002_012": tf002_012_compute,
    "tf002_013": tf002_013_compute,
    "tf002_014": tf002_014_compute,
    "tf002_015": tf002_015_compute,
}

FACTOR_ALIASES: dict[str, str] = {
    "tf002_001": "cr046",
    "tf002_002": "cr047",
    "tf002_003": "cr048",
    "tf002_004": "cr049",
    "tf002_005": "cr050",
    "tf002_006": "cr052",
    "tf002_007": "cr053",
    "tf002_008": "cr054",
    "tf002_009": "cr055",
    "tf002_010": "cr056",
    "tf002_011": "cr058",
    "tf002_012": "cr060",
    "tf002_013": "cr061",
    "tf002_014": "cr063",
    "tf002_015": "or001",
}

def aggregate_tf002(
    df_lazy: pl.LazyFrame,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
    use_aliases: bool = False,
) -> pl.LazyFrame:
    """批量计算 tf002 批次因子；未指定 names 时计算全部 15 个因子。"""
    selected = tuple(FACTOR_NAMES) if names is None else tuple(names)
    res = compute_factor_batch(
        df_lazy,
        FACTORS,
        names=selected,
        params=params,
        key_columns=KEY_COLUMNS,
    )
    if use_aliases:
        rename_map = {k: v for k, v in FACTOR_ALIASES.items() if k in res.collect_schema().names()}
        return res.rename(rename_map)
    return res


compute = aggregate_tf002

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "FACTORS",
    "aggregate_tf002",
    "compute",
]
