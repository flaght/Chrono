"""tc005 因子显式导出。"""

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

# 历史过渡期兼容别名
cr003_compute = tc005_001_compute
cr006_compute = tc005_002_compute
cr007_compute = tc005_003_compute
cr008_compute = tc005_004_compute
cr009_compute = tc005_005_compute
cr011_compute = tc005_006_compute
cr012_compute = tc005_007_compute
cr013_compute = tc005_008_compute
cr014_compute = tc005_009_compute
cr015_compute = tc005_010_compute
cr017_compute = tc005_011_compute
cr018_compute = tc005_012_compute
cr019_compute = tc005_013_compute
cr020_compute = tc005_014_compute
cr021_compute = tc005_015_compute
cr022_compute = tc005_016_compute
cr023_compute = tc005_017_compute
cr024_compute = tc005_018_compute
cr025_compute = tc005_019_compute
cr026_compute = tc005_020_compute
cr027_compute = tc005_021_compute
cr028_compute = tc005_022_compute
cr029_compute = tc005_023_compute
cr030_compute = tc005_024_compute
cr031_compute = tc005_025_compute
cr032_compute = tc005_026_compute
cr033_compute = tc005_027_compute
cr035_compute = tc005_028_compute
cr036_compute = tc005_029_compute
cr037_compute = tc005_030_compute
cr039_compute = tc005_031_compute
cr040_compute = tc005_032_compute
cr041_compute = tc005_033_compute
cr042_compute = tc005_034_compute
cr044_compute = tc005_035_compute
cr045_compute = tc005_036_compute


from .aggregator import (
    FACTOR_ALIASES,
    FACTOR_NAMES,
    FACTORS,
    aggregate_tc005,
    compute as tc005_compute_all,
)

compute_all = tc005_compute_all

__all__ = [
    "FACTOR_ALIASES",
    "FACTOR_NAMES",
    "FACTORS",
    "aggregate_tc005",
    "compute_all",
    "tc005_compute_all",
    "tc005_001_compute",
    "tc005_002_compute",
    "tc005_003_compute",
    "tc005_004_compute",
    "tc005_005_compute",
    "tc005_006_compute",
    "tc005_007_compute",
    "tc005_008_compute",
    "tc005_009_compute",
    "tc005_010_compute",
    "tc005_011_compute",
    "tc005_012_compute",
    "tc005_013_compute",
    "tc005_014_compute",
    "tc005_015_compute",
    "tc005_016_compute",
    "tc005_017_compute",
    "tc005_018_compute",
    "tc005_019_compute",
    "tc005_020_compute",
    "tc005_021_compute",
    "tc005_022_compute",
    "tc005_023_compute",
    "tc005_024_compute",
    "tc005_025_compute",
    "tc005_026_compute",
    "tc005_027_compute",
    "tc005_028_compute",
    "tc005_029_compute",
    "tc005_030_compute",
    "tc005_031_compute",
    "tc005_032_compute",
    "tc005_033_compute",
    "tc005_034_compute",
    "tc005_035_compute",
    "tc005_036_compute",
]
