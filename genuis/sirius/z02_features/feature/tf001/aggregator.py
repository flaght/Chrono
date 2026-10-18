"""tf001 期货持仓量博弈因子批次聚合引擎 (Aggregator Engine)。

负责统一组织 tf001_001 ~ tf001_044 全部 44 个期货持仓博弈因子，
提供纯 Polars Lazy 惰性计算图与批量对齐入口。
输出标准主键: ['trade_time', 'code'] 及各因子列。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import polars as pl

from feature.utils.common import KEY_COLUMNS, FactorCompute, compute_factor_batch

from .tf001_001 import compute as tf001_001_compute
from .tf001_002 import compute as tf001_002_compute
from .tf001_003 import compute as tf001_003_compute
from .tf001_004 import compute as tf001_004_compute
from .tf001_005 import compute as tf001_005_compute
from .tf001_006 import compute as tf001_006_compute
from .tf001_007 import compute as tf001_007_compute
from .tf001_008 import compute as tf001_008_compute
from .tf001_009 import compute as tf001_009_compute
from .tf001_010 import compute as tf001_010_compute
from .tf001_011 import compute as tf001_011_compute
from .tf001_012 import compute as tf001_012_compute
from .tf001_013 import compute as tf001_013_compute
from .tf001_014 import compute as tf001_014_compute
from .tf001_015 import compute as tf001_015_compute
from .tf001_016 import compute as tf001_016_compute
from .tf001_017 import compute as tf001_017_compute
from .tf001_018 import compute as tf001_018_compute
from .tf001_019 import compute as tf001_019_compute
from .tf001_020 import compute as tf001_020_compute
from .tf001_021 import compute as tf001_021_compute
from .tf001_022 import compute as tf001_022_compute
from .tf001_023 import compute as tf001_023_compute
from .tf001_024 import compute as tf001_024_compute
from .tf001_025 import compute as tf001_025_compute
from .tf001_026 import compute as tf001_026_compute
from .tf001_027 import compute as tf001_027_compute
from .tf001_028 import compute as tf001_028_compute
from .tf001_029 import compute as tf001_029_compute
from .tf001_030 import compute as tf001_030_compute
from .tf001_031 import compute as tf001_031_compute
from .tf001_032 import compute as tf001_032_compute
from .tf001_033 import compute as tf001_033_compute
from .tf001_034 import compute as tf001_034_compute
from .tf001_035 import compute as tf001_035_compute
from .tf001_036 import compute as tf001_036_compute
from .tf001_037 import compute as tf001_037_compute
from .tf001_038 import compute as tf001_038_compute
from .tf001_039 import compute as tf001_039_compute
from .tf001_040 import compute as tf001_040_compute
from .tf001_041 import compute as tf001_041_compute
from .tf001_042 import compute as tf001_042_compute
from .tf001_043 import compute as tf001_043_compute
from .tf001_044 import compute as tf001_044_compute

FACTOR_NAMES: list[str] = [f"tf001_{i:03d}" for i in range(1, 45)]

FACTORS: dict[str, FactorCompute] = {
    "tf001_001": tf001_001_compute,
    "tf001_002": tf001_002_compute,
    "tf001_003": tf001_003_compute,
    "tf001_004": tf001_004_compute,
    "tf001_005": tf001_005_compute,
    "tf001_006": tf001_006_compute,
    "tf001_007": tf001_007_compute,
    "tf001_008": tf001_008_compute,
    "tf001_009": tf001_009_compute,
    "tf001_010": tf001_010_compute,
    "tf001_011": tf001_011_compute,
    "tf001_012": tf001_012_compute,
    "tf001_013": tf001_013_compute,
    "tf001_014": tf001_014_compute,
    "tf001_015": tf001_015_compute,
    "tf001_016": tf001_016_compute,
    "tf001_017": tf001_017_compute,
    "tf001_018": tf001_018_compute,
    "tf001_019": tf001_019_compute,
    "tf001_020": tf001_020_compute,
    "tf001_021": tf001_021_compute,
    "tf001_022": tf001_022_compute,
    "tf001_023": tf001_023_compute,
    "tf001_024": tf001_024_compute,
    "tf001_025": tf001_025_compute,
    "tf001_026": tf001_026_compute,
    "tf001_027": tf001_027_compute,
    "tf001_028": tf001_028_compute,
    "tf001_029": tf001_029_compute,
    "tf001_030": tf001_030_compute,
    "tf001_031": tf001_031_compute,
    "tf001_032": tf001_032_compute,
    "tf001_033": tf001_033_compute,
    "tf001_034": tf001_034_compute,
    "tf001_035": tf001_035_compute,
    "tf001_036": tf001_036_compute,
    "tf001_037": tf001_037_compute,
    "tf001_038": tf001_038_compute,
    "tf001_039": tf001_039_compute,
    "tf001_040": tf001_040_compute,
    "tf001_041": tf001_041_compute,
    "tf001_042": tf001_042_compute,
    "tf001_043": tf001_043_compute,
    "tf001_044": tf001_044_compute,
}

FACTOR_ALIASES: dict[str, str] = {
    "tf001_001": "oi001",
    "tf001_002": "oi002",
    "tf001_003": "oi003",
    "tf001_004": "oi004",
    "tf001_005": "oi005",
    "tf001_006": "oi006",
    "tf001_007": "oi008",
    "tf001_008": "oi009",
    "tf001_009": "oi010",
    "tf001_010": "oi011",
    "tf001_011": "oi012",
    "tf001_012": "oi013",
    "tf001_013": "oi014",
    "tf001_014": "oi015",
    "tf001_015": "oi016",
    "tf001_016": "oi017",
    "tf001_017": "oi018",
    "tf001_018": "oi019",
    "tf001_019": "oi020",
    "tf001_020": "oi021",
    "tf001_021": "oi022",
    "tf001_022": "oi023",
    "tf001_023": "oi024",
    "tf001_024": "oi025",
    "tf001_025": "oi026",
    "tf001_026": "oi027",
    "tf001_027": "oi028",
    "tf001_028": "oi029",
    "tf001_029": "oi030",
    "tf001_030": "oi031",
    "tf001_031": "oi032",
    "tf001_032": "oi033",
    "tf001_033": "oi034",
    "tf001_034": "oi035",
    "tf001_035": "oi036",
    "tf001_036": "oi037",
    "tf001_037": "oi038",
    "tf001_038": "oi039",
    "tf001_039": "oi040",
    "tf001_040": "oi041",
    "tf001_041": "oi042",
    "tf001_042": "oi043",
    "tf001_043": "oi045",
    "tf001_044": "oi046",
}

def aggregate_tf001(
    df_lazy: pl.LazyFrame,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
    use_aliases: bool = False,
) -> pl.LazyFrame:
    """批量计算 tf001 批次因子；未指定 names 时计算全部 44 个因子。"""
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


compute = aggregate_tf001

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_ALIASES",
    "FACTORS",
    "aggregate_tf001",
    "compute",
]
