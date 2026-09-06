"""tc005 通用技术形态与时钟周期类因子批次聚合引擎 (Aggregator Engine)。

负责统一组织 tc005_001 ~ tc005_036 全部 36 个技术因子，
提供纯 Polars Lazy 惰性计算图与批量对齐入口。
输出标准主键: ['trade_time', 'code'] 及各因子列。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import polars as pl

from feature.utils.common import KEY_COLUMNS, FactorCompute, compute_factor_batch

from .tc005_001 import compute as tc005_001_compute
from .tc005_002 import compute as tc005_002_compute
from .tc005_003 import compute as tc005_003_compute
from .tc005_004 import compute as tc005_004_compute
from .tc005_005 import compute as tc005_005_compute
from .tc005_006 import compute as tc005_006_compute
from .tc005_007 import compute as tc005_007_compute
from .tc005_008 import compute as tc005_008_compute
from .tc005_009 import compute as tc005_009_compute
from .tc005_010 import compute as tc005_010_compute
from .tc005_011 import compute as tc005_011_compute
from .tc005_012 import compute as tc005_012_compute
from .tc005_013 import compute as tc005_013_compute
from .tc005_014 import compute as tc005_014_compute
from .tc005_015 import compute as tc005_015_compute
from .tc005_016 import compute as tc005_016_compute
from .tc005_017 import compute as tc005_017_compute
from .tc005_018 import compute as tc005_018_compute
from .tc005_019 import compute as tc005_019_compute
from .tc005_020 import compute as tc005_020_compute
from .tc005_021 import compute as tc005_021_compute
from .tc005_022 import compute as tc005_022_compute
from .tc005_023 import compute as tc005_023_compute
from .tc005_024 import compute as tc005_024_compute
from .tc005_025 import compute as tc005_025_compute
from .tc005_026 import compute as tc005_026_compute
from .tc005_027 import compute as tc005_027_compute
from .tc005_028 import compute as tc005_028_compute
from .tc005_029 import compute as tc005_029_compute
from .tc005_030 import compute as tc005_030_compute
from .tc005_031 import compute as tc005_031_compute
from .tc005_032 import compute as tc005_032_compute
from .tc005_033 import compute as tc005_033_compute
from .tc005_034 import compute as tc005_034_compute
from .tc005_035 import compute as tc005_035_compute
from .tc005_036 import compute as tc005_036_compute

FACTOR_NAMES: list[str] = [f"tc005_{i:03d}" for i in range(1, 37)]

FACTORS: dict[str, FactorCompute] = {
    "tc005_001": tc005_001_compute,
    "tc005_002": tc005_002_compute,
    "tc005_003": tc005_003_compute,
    "tc005_004": tc005_004_compute,
    "tc005_005": tc005_005_compute,
    "tc005_006": tc005_006_compute,
    "tc005_007": tc005_007_compute,
    "tc005_008": tc005_008_compute,
    "tc005_009": tc005_009_compute,
    "tc005_010": tc005_010_compute,
    "tc005_011": tc005_011_compute,
    "tc005_012": tc005_012_compute,
    "tc005_013": tc005_013_compute,
    "tc005_014": tc005_014_compute,
    "tc005_015": tc005_015_compute,
    "tc005_016": tc005_016_compute,
    "tc005_017": tc005_017_compute,
    "tc005_018": tc005_018_compute,
    "tc005_019": tc005_019_compute,
    "tc005_020": tc005_020_compute,
    "tc005_021": tc005_021_compute,
    "tc005_022": tc005_022_compute,
    "tc005_023": tc005_023_compute,
    "tc005_024": tc005_024_compute,
    "tc005_025": tc005_025_compute,
    "tc005_026": tc005_026_compute,
    "tc005_027": tc005_027_compute,
    "tc005_028": tc005_028_compute,
    "tc005_029": tc005_029_compute,
    "tc005_030": tc005_030_compute,
    "tc005_031": tc005_031_compute,
    "tc005_032": tc005_032_compute,
    "tc005_033": tc005_033_compute,
    "tc005_034": tc005_034_compute,
    "tc005_035": tc005_035_compute,
    "tc005_036": tc005_036_compute,
}

FACTOR_ALIASES: dict[str, str] = {
    "tc005_001": "cr003",
    "tc005_002": "cr006",
    "tc005_003": "cr007",
    "tc005_004": "cr008",
    "tc005_005": "cr009",
    "tc005_006": "cr011",
    "tc005_007": "cr012",
    "tc005_008": "cr013",
    "tc005_009": "cr014",
    "tc005_010": "cr015",
    "tc005_011": "cr017",
    "tc005_012": "cr018",
    "tc005_013": "cr019",
    "tc005_014": "cr020",
    "tc005_015": "cr021",
    "tc005_016": "cr022",
    "tc005_017": "cr023",
    "tc005_018": "cr024",
    "tc005_019": "cr025",
    "tc005_020": "cr026",
    "tc005_021": "cr027",
    "tc005_022": "cr028",
    "tc005_023": "cr029",
    "tc005_024": "cr030",
    "tc005_025": "cr031",
    "tc005_026": "cr032",
    "tc005_027": "cr033",
    "tc005_028": "cr035",
    "tc005_029": "cr036",
    "tc005_030": "cr037",
    "tc005_031": "cr039",
    "tc005_032": "cr040",
    "tc005_033": "cr041",
    "tc005_034": "cr042",
    "tc005_035": "cr044",
    "tc005_036": "cr045",
}

def aggregate_tc005(
    df_lazy: pl.LazyFrame,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
    use_aliases: bool = False,
) -> pl.LazyFrame:
    """批量计算 tc005 批次因子；未指定 names 时计算全部 36 个因子。"""
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


compute = aggregate_tc005

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "FACTORS",
    "aggregate_tc005",
    "compute",
]
