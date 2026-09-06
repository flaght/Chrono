"""tc002 通用技术量价关系与振荡器类因子批次聚合引擎 (Aggregator Engine)。

负责统一组织 tc002_001 ~ tc002_038 全部 38 个技术因子，
提供纯 Polars Lazy 惰性计算图与批量对齐入口。
输出标准主键: ['trade_time', 'code'] 及各因子列。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import polars as pl

from feature.utils.common import KEY_COLUMNS, FactorCompute, compute_factor_batch

from .tc002_001 import compute as tc002_001_compute
from .tc002_002 import compute as tc002_002_compute
from .tc002_003 import compute as tc002_003_compute
from .tc002_004 import compute as tc002_004_compute
from .tc002_005 import compute as tc002_005_compute
from .tc002_006 import compute as tc002_006_compute
from .tc002_007 import compute as tc002_007_compute
from .tc002_008 import compute as tc002_008_compute
from .tc002_009 import compute as tc002_009_compute
from .tc002_010 import compute as tc002_010_compute
from .tc002_011 import compute as tc002_011_compute
from .tc002_012 import compute as tc002_012_compute
from .tc002_013 import compute as tc002_013_compute
from .tc002_014 import compute as tc002_014_compute
from .tc002_015 import compute as tc002_015_compute
from .tc002_016 import compute as tc002_016_compute
from .tc002_017 import compute as tc002_017_compute
from .tc002_018 import compute as tc002_018_compute
from .tc002_019 import compute as tc002_019_compute
from .tc002_020 import compute as tc002_020_compute
from .tc002_021 import compute as tc002_021_compute
from .tc002_022 import compute as tc002_022_compute
from .tc002_023 import compute as tc002_023_compute
from .tc002_024 import compute as tc002_024_compute
from .tc002_025 import compute as tc002_025_compute
from .tc002_026 import compute as tc002_026_compute
from .tc002_027 import compute as tc002_027_compute
from .tc002_028 import compute as tc002_028_compute
from .tc002_029 import compute as tc002_029_compute
from .tc002_030 import compute as tc002_030_compute
from .tc002_031 import compute as tc002_031_compute
from .tc002_032 import compute as tc002_032_compute
from .tc002_033 import compute as tc002_033_compute
from .tc002_034 import compute as tc002_034_compute
from .tc002_035 import compute as tc002_035_compute
from .tc002_036 import compute as tc002_036_compute
from .tc002_037 import compute as tc002_037_compute
from .tc002_038 import compute as tc002_038_compute

FACTOR_NAMES: list[str] = [f"tc002_{i:03d}" for i in range(1, 39)]

FACTORS: dict[str, FactorCompute] = {
    "tc002_001": tc002_001_compute,
    "tc002_002": tc002_002_compute,
    "tc002_003": tc002_003_compute,
    "tc002_004": tc002_004_compute,
    "tc002_005": tc002_005_compute,
    "tc002_006": tc002_006_compute,
    "tc002_007": tc002_007_compute,
    "tc002_008": tc002_008_compute,
    "tc002_009": tc002_009_compute,
    "tc002_010": tc002_010_compute,
    "tc002_011": tc002_011_compute,
    "tc002_012": tc002_012_compute,
    "tc002_013": tc002_013_compute,
    "tc002_014": tc002_014_compute,
    "tc002_015": tc002_015_compute,
    "tc002_016": tc002_016_compute,
    "tc002_017": tc002_017_compute,
    "tc002_018": tc002_018_compute,
    "tc002_019": tc002_019_compute,
    "tc002_020": tc002_020_compute,
    "tc002_021": tc002_021_compute,
    "tc002_022": tc002_022_compute,
    "tc002_023": tc002_023_compute,
    "tc002_024": tc002_024_compute,
    "tc002_025": tc002_025_compute,
    "tc002_026": tc002_026_compute,
    "tc002_027": tc002_027_compute,
    "tc002_028": tc002_028_compute,
    "tc002_029": tc002_029_compute,
    "tc002_030": tc002_030_compute,
    "tc002_031": tc002_031_compute,
    "tc002_032": tc002_032_compute,
    "tc002_033": tc002_033_compute,
    "tc002_034": tc002_034_compute,
    "tc002_035": tc002_035_compute,
    "tc002_036": tc002_036_compute,
    "tc002_037": tc002_037_compute,
    "tc002_038": tc002_038_compute,
}

FACTOR_ALIASES: dict[str, str] = {
    "tc002_001": "cj002",
    "tc002_002": "cj003",
    "tc002_003": "cj006",
    "tc002_004": "cj007",
    "tc002_005": "cj009",
    "tc002_006": "cj010",
    "tc002_007": "cj011",
    "tc002_008": "cj013",
    "tc002_009": "cj014",
    "tc002_010": "cj015",
    "tc002_011": "cj016",
    "tc002_012": "db001",
    "tc002_013": "db002",
    "tc002_014": "db003",
    "tc002_015": "db004",
    "tc002_016": "db005",
    "tc002_017": "db006",
    "tc002_018": "db007",
    "tc002_019": "ixy001",
    "tc002_020": "ixy002",
    "tc002_021": "ixy003",
    "tc002_022": "ixy004",
    "tc002_023": "ixy005",
    "tc002_024": "ixy006",
    "tc002_025": "ixy007",
    "tc002_026": "ixy008",
    "tc002_027": "ixy009",
    "tc002_028": "ixy010",
    "tc002_029": "ixy012",
    "tc002_030": "ixy013",
    "tc002_031": "ixy014",
    "tc002_032": "ixy015",
    "tc002_033": "ixy016",
    "tc002_034": "xy001",
    "tc002_035": "xy002",
    "tc002_036": "xy003",
    "tc002_037": "xy004",
    "tc002_038": "xy005",
}

def aggregate_tc002(
    df_lazy: pl.LazyFrame,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
    use_aliases: bool = False,
) -> pl.LazyFrame:
    """批量计算 tc002 批次因子；未指定 names 时计算全部 38 个因子。"""
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


compute = aggregate_tc002

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "FACTORS",
    "aggregate_tc002",
    "compute",
]
