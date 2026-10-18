"""tc002 因子显式导出。"""

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

# 历史过渡期兼容别名
cj002_compute = tc002_001_compute
cj003_compute = tc002_002_compute
cj006_compute = tc002_003_compute
cj007_compute = tc002_004_compute
cj009_compute = tc002_005_compute
cj010_compute = tc002_006_compute
cj011_compute = tc002_007_compute
cj013_compute = tc002_008_compute
cj014_compute = tc002_009_compute
cj015_compute = tc002_010_compute
cj016_compute = tc002_011_compute
db001_compute = tc002_012_compute
db002_compute = tc002_013_compute
db003_compute = tc002_014_compute
db004_compute = tc002_015_compute
db005_compute = tc002_016_compute
db006_compute = tc002_017_compute
db007_compute = tc002_018_compute
ixy001_compute = tc002_019_compute
ixy002_compute = tc002_020_compute
ixy003_compute = tc002_021_compute
ixy004_compute = tc002_022_compute
ixy005_compute = tc002_023_compute
ixy006_compute = tc002_024_compute
ixy007_compute = tc002_025_compute
ixy008_compute = tc002_026_compute
ixy009_compute = tc002_027_compute
ixy010_compute = tc002_028_compute
ixy012_compute = tc002_029_compute
ixy013_compute = tc002_030_compute
ixy014_compute = tc002_031_compute
ixy015_compute = tc002_032_compute
ixy016_compute = tc002_033_compute
xy001_compute = tc002_034_compute
xy002_compute = tc002_035_compute
xy003_compute = tc002_036_compute
xy004_compute = tc002_037_compute
xy005_compute = tc002_038_compute


from .aggregator import (
    FACTOR_ALIASES,
    FACTOR_NAMES,
    FACTORS,
    aggregate_tc002,
    compute as tc002_compute_all,
)

compute_all = tc002_compute_all

__all__ = [
    "FACTOR_ALIASES",
    "FACTOR_NAMES",
    "FACTORS",
    "aggregate_tc002",
    "compute_all",
    "tc002_compute_all",
    "tc002_001_compute",
    "tc002_002_compute",
    "tc002_003_compute",
    "tc002_004_compute",
    "tc002_005_compute",
    "tc002_006_compute",
    "tc002_007_compute",
    "tc002_008_compute",
    "tc002_009_compute",
    "tc002_010_compute",
    "tc002_011_compute",
    "tc002_012_compute",
    "tc002_013_compute",
    "tc002_014_compute",
    "tc002_015_compute",
    "tc002_016_compute",
    "tc002_017_compute",
    "tc002_018_compute",
    "tc002_019_compute",
    "tc002_020_compute",
    "tc002_021_compute",
    "tc002_022_compute",
    "tc002_023_compute",
    "tc002_024_compute",
    "tc002_025_compute",
    "tc002_026_compute",
    "tc002_027_compute",
    "tc002_028_compute",
    "tc002_029_compute",
    "tc002_030_compute",
    "tc002_031_compute",
    "tc002_032_compute",
    "tc002_033_compute",
    "tc002_034_compute",
    "tc002_035_compute",
    "tc002_036_compute",
    "tc002_037_compute",
    "tc002_038_compute",
]
