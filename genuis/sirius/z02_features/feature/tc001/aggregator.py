"""tc001 通用技术动量与趋势因子批次聚合引擎 (Aggregator Engine)。

负责统一组织 tc001_001 ~ tc001_038 全部 38 个技术动量与趋势因子，
提供纯 Polars Lazy 惰性计算图与批量对齐入口。
输出标准主键: ['trade_time', 'code'] 及各因子列。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import polars as pl

from feature.utils.common import KEY_COLUMNS, FactorCompute, compute_factor_batch

from .tc001_001 import compute as tc001_001_compute
from .tc001_002 import compute as tc001_002_compute
from .tc001_003 import compute as tc001_003_compute
from .tc001_004 import compute as tc001_004_compute
from .tc001_005 import compute as tc001_005_compute
from .tc001_006 import compute as tc001_006_compute
from .tc001_007 import compute as tc001_007_compute
from .tc001_008 import compute as tc001_008_compute
from .tc001_009 import compute as tc001_009_compute
from .tc001_010 import compute as tc001_010_compute
from .tc001_011 import compute as tc001_011_compute
from .tc001_012 import compute as tc001_012_compute
from .tc001_013 import compute as tc001_013_compute
from .tc001_014 import compute as tc001_014_compute
from .tc001_015 import compute as tc001_015_compute
from .tc001_016 import compute as tc001_016_compute
from .tc001_017 import compute as tc001_017_compute
from .tc001_018 import compute as tc001_018_compute
from .tc001_019 import compute as tc001_019_compute
from .tc001_020 import compute as tc001_020_compute
from .tc001_021 import compute as tc001_021_compute
from .tc001_022 import compute as tc001_022_compute
from .tc001_023 import compute as tc001_023_compute
from .tc001_024 import compute as tc001_024_compute
from .tc001_025 import compute as tc001_025_compute
from .tc001_026 import compute as tc001_026_compute
from .tc001_027 import compute as tc001_027_compute
from .tc001_028 import compute as tc001_028_compute
from .tc001_029 import compute as tc001_029_compute
from .tc001_030 import compute as tc001_030_compute
from .tc001_031 import compute as tc001_031_compute
from .tc001_032 import compute as tc001_032_compute
from .tc001_033 import compute as tc001_033_compute
from .tc001_034 import compute as tc001_034_compute
from .tc001_035 import compute as tc001_035_compute
from .tc001_036 import compute as tc001_036_compute
from .tc001_037 import compute as tc001_037_compute
from .tc001_038 import compute as tc001_038_compute

FACTOR_NAMES: list[str] = [f"tc001_{i:03d}" for i in range(1, 39)]

FACTORS: dict[str, FactorCompute] = {
    "tc001_001": tc001_001_compute,
    "tc001_002": tc001_002_compute,
    "tc001_003": tc001_003_compute,
    "tc001_004": tc001_004_compute,
    "tc001_005": tc001_005_compute,
    "tc001_006": tc001_006_compute,
    "tc001_007": tc001_007_compute,
    "tc001_008": tc001_008_compute,
    "tc001_009": tc001_009_compute,
    "tc001_010": tc001_010_compute,
    "tc001_011": tc001_011_compute,
    "tc001_012": tc001_012_compute,
    "tc001_013": tc001_013_compute,
    "tc001_014": tc001_014_compute,
    "tc001_015": tc001_015_compute,
    "tc001_016": tc001_016_compute,
    "tc001_017": tc001_017_compute,
    "tc001_018": tc001_018_compute,
    "tc001_019": tc001_019_compute,
    "tc001_020": tc001_020_compute,
    "tc001_021": tc001_021_compute,
    "tc001_022": tc001_022_compute,
    "tc001_023": tc001_023_compute,
    "tc001_024": tc001_024_compute,
    "tc001_025": tc001_025_compute,
    "tc001_026": tc001_026_compute,
    "tc001_027": tc001_027_compute,
    "tc001_028": tc001_028_compute,
    "tc001_029": tc001_029_compute,
    "tc001_030": tc001_030_compute,
    "tc001_031": tc001_031_compute,
    "tc001_032": tc001_032_compute,
    "tc001_033": tc001_033_compute,
    "tc001_034": tc001_034_compute,
    "tc001_035": tc001_035_compute,
    "tc001_036": tc001_036_compute,
    "tc001_037": tc001_037_compute,
    "tc001_038": tc001_038_compute,
}

FACTOR_ALIASES: dict[str, str] = {
    "tc001_001": "ta001",
    "tc001_002": "ta002",
    "tc001_003": "ta003",
    "tc001_004": "ta004",
    "tc001_005": "ta005",
    "tc001_006": "ta006",
    "tc001_007": "ta007",
    "tc001_008": "ta008",
    "tc001_009": "ta009",
    "tc001_010": "ta010",
    "tc001_011": "ta011",
    "tc001_012": "ta012",
    "tc001_013": "ta013",
    "tc001_014": "ta014",
    "tc001_015": "ta015",
    "tc001_016": "ta016",
    "tc001_017": "ta017",
    "tc001_018": "ta018",
    "tc001_019": "ta019",
    "tc001_020": "ta020",
    "tc001_021": "ta021",
    "tc001_022": "ta022",
    "tc001_023": "ta023",
    "tc001_024": "ta024",
    "tc001_025": "ta025",
    "tc001_026": "ta026",
    "tc001_027": "ta027",
    "tc001_028": "ta028",
    "tc001_029": "ta029",
    "tc001_030": "ta030",
    "tc001_031": "ta031",
    "tc001_032": "ta032",
    "tc001_033": "ta033",
    "tc001_034": "ta034",
    "tc001_035": "ta035",
    "tc001_036": "ta036",
    "tc001_037": "ta037",
    "tc001_038": "ta038",
    "tc001_023_12_26": "ta023_12_26",
    "tc001_025_5": "ta025_5",
    "tc001_025_10": "ta025_10",
}

def aggregate_tc001(
    df_lazy: pl.LazyFrame,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
    use_aliases: bool = False,
) -> pl.LazyFrame:
    """批量计算 tc001 批次因子；未指定 names 时计算全部 38 个因子。"""
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


compute = aggregate_tc001

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "FACTORS",
    "aggregate_tc001",
    "compute",
]
