"""tc006 通用技术统计矩与高阶特征类因子批次聚合引擎 (Aggregator Engine)。

负责统一组织 tc006_001 ~ tc006_055 全部 55 个技术因子，
提供纯 Polars Lazy 惰性计算图与批量对齐入口。
输出标准主键: ['trade_time', 'code'] 及各因子列。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import polars as pl

from feature.utils.common import KEY_COLUMNS, FactorCompute, compute_factor_batch

from .tc006_001 import compute as tc006_001_compute
from .tc006_002 import compute as tc006_002_compute
from .tc006_003 import compute as tc006_003_compute
from .tc006_004 import compute as tc006_004_compute
from .tc006_005 import compute as tc006_005_compute
from .tc006_006 import compute as tc006_006_compute
from .tc006_007 import compute as tc006_007_compute
from .tc006_008 import compute as tc006_008_compute
from .tc006_009 import compute as tc006_009_compute
from .tc006_010 import compute as tc006_010_compute
from .tc006_011 import compute as tc006_011_compute
from .tc006_012 import compute as tc006_012_compute
from .tc006_013 import compute as tc006_013_compute
from .tc006_014 import compute as tc006_014_compute
from .tc006_015 import compute as tc006_015_compute
from .tc006_016 import compute as tc006_016_compute
from .tc006_017 import compute as tc006_017_compute
from .tc006_018 import compute as tc006_018_compute
from .tc006_019 import compute as tc006_019_compute
from .tc006_020 import compute as tc006_020_compute
from .tc006_021 import compute as tc006_021_compute
from .tc006_022 import compute as tc006_022_compute
from .tc006_023 import compute as tc006_023_compute
from .tc006_024 import compute as tc006_024_compute
from .tc006_025 import compute as tc006_025_compute
from .tc006_026 import compute as tc006_026_compute
from .tc006_027 import compute as tc006_027_compute
from .tc006_028 import compute as tc006_028_compute
from .tc006_029 import compute as tc006_029_compute
from .tc006_030 import compute as tc006_030_compute
from .tc006_031 import compute as tc006_031_compute
from .tc006_032 import compute as tc006_032_compute
from .tc006_033 import compute as tc006_033_compute
from .tc006_034 import compute as tc006_034_compute
from .tc006_035 import compute as tc006_035_compute
from .tc006_036 import compute as tc006_036_compute
from .tc006_037 import compute as tc006_037_compute
from .tc006_038 import compute as tc006_038_compute
from .tc006_039 import compute as tc006_039_compute
from .tc006_040 import compute as tc006_040_compute
from .tc006_041 import compute as tc006_041_compute
from .tc006_042 import compute as tc006_042_compute
from .tc006_043 import compute as tc006_043_compute
from .tc006_044 import compute as tc006_044_compute
from .tc006_045 import compute as tc006_045_compute
from .tc006_046 import compute as tc006_046_compute
from .tc006_047 import compute as tc006_047_compute
from .tc006_048 import compute as tc006_048_compute
from .tc006_049 import compute as tc006_049_compute
from .tc006_050 import compute as tc006_050_compute
from .tc006_051 import compute as tc006_051_compute
from .tc006_052 import compute as tc006_052_compute
from .tc006_053 import compute as tc006_053_compute
from .tc006_054 import compute as tc006_054_compute
from .tc006_055 import compute as tc006_055_compute

FACTOR_NAMES: list[str] = [f"tc006_{i:03d}" for i in range(1, 56)]

FACTORS: dict[str, FactorCompute] = {
    "tc006_001": tc006_001_compute,
    "tc006_002": tc006_002_compute,
    "tc006_003": tc006_003_compute,
    "tc006_004": tc006_004_compute,
    "tc006_005": tc006_005_compute,
    "tc006_006": tc006_006_compute,
    "tc006_007": tc006_007_compute,
    "tc006_008": tc006_008_compute,
    "tc006_009": tc006_009_compute,
    "tc006_010": tc006_010_compute,
    "tc006_011": tc006_011_compute,
    "tc006_012": tc006_012_compute,
    "tc006_013": tc006_013_compute,
    "tc006_014": tc006_014_compute,
    "tc006_015": tc006_015_compute,
    "tc006_016": tc006_016_compute,
    "tc006_017": tc006_017_compute,
    "tc006_018": tc006_018_compute,
    "tc006_019": tc006_019_compute,
    "tc006_020": tc006_020_compute,
    "tc006_021": tc006_021_compute,
    "tc006_022": tc006_022_compute,
    "tc006_023": tc006_023_compute,
    "tc006_024": tc006_024_compute,
    "tc006_025": tc006_025_compute,
    "tc006_026": tc006_026_compute,
    "tc006_027": tc006_027_compute,
    "tc006_028": tc006_028_compute,
    "tc006_029": tc006_029_compute,
    "tc006_030": tc006_030_compute,
    "tc006_031": tc006_031_compute,
    "tc006_032": tc006_032_compute,
    "tc006_033": tc006_033_compute,
    "tc006_034": tc006_034_compute,
    "tc006_035": tc006_035_compute,
    "tc006_036": tc006_036_compute,
    "tc006_037": tc006_037_compute,
    "tc006_038": tc006_038_compute,
    "tc006_039": tc006_039_compute,
    "tc006_040": tc006_040_compute,
    "tc006_041": tc006_041_compute,
    "tc006_042": tc006_042_compute,
    "tc006_043": tc006_043_compute,
    "tc006_044": tc006_044_compute,
    "tc006_045": tc006_045_compute,
    "tc006_046": tc006_046_compute,
    "tc006_047": tc006_047_compute,
    "tc006_048": tc006_048_compute,
    "tc006_049": tc006_049_compute,
    "tc006_050": tc006_050_compute,
    "tc006_051": tc006_051_compute,
    "tc006_052": tc006_052_compute,
    "tc006_053": tc006_053_compute,
    "tc006_054": tc006_054_compute,
    "tc006_055": tc006_055_compute,
}

FACTOR_ALIASES: dict[str, str] = {
    "tc006_001": "tb001",
    "tc006_002": "tb002",
    "tc006_003": "tb003",
    "tc006_004": "tb004",
    "tc006_005": "tb005",
    "tc006_006": "tb006",
    "tc006_007": "tb007",
    "tc006_008": "tb008",
    "tc006_009": "tb009",
    "tc006_010": "tb010",
    "tc006_011": "tb011",
    "tc006_012": "tb012",
    "tc006_013": "tb013",
    "tc006_014": "tb014",
    "tc006_015": "tb015",
    "tc006_016": "tb016",
    "tc006_017": "tb017",
    "tc006_018": "tb018",
    "tc006_019": "tb019",
    "tc006_020": "tb020",
    "tc006_021": "tb021",
    "tc006_022": "tb022",
    "tc006_023": "tb023",
    "tc006_024": "tb024",
    "tc006_025": "tb025",
    "tc006_026": "tb026",
    "tc006_027": "tb027",
    "tc006_028": "tb028",
    "tc006_029": "tb029",
    "tc006_030": "tb030",
    "tc006_031": "tb031",
    "tc006_032": "tb032",
    "tc006_033": "tb033",
    "tc006_034": "tb034",
    "tc006_035": "tb035",
    "tc006_036": "tb036",
    "tc006_037": "tb037",
    "tc006_038": "tb038",
    "tc006_039": "tb039",
    "tc006_040": "tb040",
    "tc006_041": "tb041",
    "tc006_042": "tb042",
    "tc006_043": "tb043",
    "tc006_044": "tb044",
    "tc006_045": "tb045",
    "tc006_046": "tb046",
    "tc006_047": "tb047",
    "tc006_048": "tb048",
    "tc006_049": "tb049",
    "tc006_050": "tb050",
    "tc006_051": "tb051",
    "tc006_052": "tb052",
    "tc006_053": "tb053",
    "tc006_054": "tb054",
    "tc006_055": "tb055",
    "tc006_001_7_21": "tb001_7_21",
    "tc006_001_12_26": "tb001_12_26",
}

def aggregate_tc006(
    df_lazy: pl.LazyFrame,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
    use_aliases: bool = False,
) -> pl.LazyFrame:
    """批量计算 tc006 批次因子；未指定 names 时计算全部 55 个因子。"""
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


compute = aggregate_tc006

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "FACTORS",
    "aggregate_tc006",
    "compute",
]
