"""tc001 因子显式导出。"""

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

# 历史过渡期兼容别名
ta001_compute = tc001_001_compute
ta002_compute = tc001_002_compute
ta003_compute = tc001_003_compute
ta004_compute = tc001_004_compute
ta005_compute = tc001_005_compute
ta006_compute = tc001_006_compute
ta007_compute = tc001_007_compute
ta008_compute = tc001_008_compute
ta009_compute = tc001_009_compute
ta010_compute = tc001_010_compute
ta011_compute = tc001_011_compute
ta012_compute = tc001_012_compute
ta013_compute = tc001_013_compute
ta014_compute = tc001_014_compute
ta015_compute = tc001_015_compute
ta016_compute = tc001_016_compute
ta017_compute = tc001_017_compute
ta018_compute = tc001_018_compute
ta019_compute = tc001_019_compute
ta020_compute = tc001_020_compute
ta021_compute = tc001_021_compute
ta022_compute = tc001_022_compute
ta023_compute = tc001_023_compute
ta024_compute = tc001_024_compute
ta025_compute = tc001_025_compute
ta026_compute = tc001_026_compute
ta027_compute = tc001_027_compute
ta028_compute = tc001_028_compute
ta029_compute = tc001_029_compute
ta030_compute = tc001_030_compute
ta031_compute = tc001_031_compute
ta032_compute = tc001_032_compute
ta033_compute = tc001_033_compute
ta034_compute = tc001_034_compute
ta035_compute = tc001_035_compute
ta036_compute = tc001_036_compute
ta037_compute = tc001_037_compute
ta038_compute = tc001_038_compute

from .aggregator import (
    FACTOR_ALIASES,
    FACTOR_NAMES,
    FACTORS,
    aggregate_tc001,
    compute as tc001_compute_all,
)

compute_all = tc001_compute_all

__all__ = [
    "FACTOR_ALIASES",
    "FACTOR_NAMES",
    "FACTORS",
    "aggregate_tc001",
    "compute_all",
    "tc001_compute_all",
    "tc001_001_compute",
    "tc001_002_compute",
    "tc001_003_compute",
    "tc001_004_compute",
    "tc001_005_compute",
    "tc001_006_compute",
    "tc001_007_compute",
    "tc001_008_compute",
    "tc001_009_compute",
    "tc001_010_compute",
    "tc001_011_compute",
    "tc001_012_compute",
    "tc001_013_compute",
    "tc001_014_compute",
    "tc001_015_compute",
    "tc001_016_compute",
    "tc001_017_compute",
    "tc001_018_compute",
    "tc001_019_compute",
    "tc001_020_compute",
    "tc001_021_compute",
    "tc001_022_compute",
    "tc001_023_compute",
    "tc001_024_compute",
    "tc001_025_compute",
    "tc001_026_compute",
    "tc001_027_compute",
    "tc001_028_compute",
    "tc001_029_compute",
    "tc001_030_compute",
    "tc001_031_compute",
    "tc001_032_compute",
    "tc001_033_compute",
    "tc001_034_compute",
    "tc001_035_compute",
    "tc001_036_compute",
    "tc001_037_compute",
    "tc001_038_compute",
]

