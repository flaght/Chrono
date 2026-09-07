"""tc006 因子显式导出。"""

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

# 历史过渡期兼容别名
tb001_compute = tc006_001_compute
tb002_compute = tc006_002_compute
tb003_compute = tc006_003_compute
tb004_compute = tc006_004_compute
tb005_compute = tc006_005_compute
tb006_compute = tc006_006_compute
tb007_compute = tc006_007_compute
tb008_compute = tc006_008_compute
tb009_compute = tc006_009_compute
tb010_compute = tc006_010_compute
tb011_compute = tc006_011_compute
tb012_compute = tc006_012_compute
tb013_compute = tc006_013_compute
tb014_compute = tc006_014_compute
tb015_compute = tc006_015_compute
tb016_compute = tc006_016_compute
tb017_compute = tc006_017_compute
tb018_compute = tc006_018_compute
tb019_compute = tc006_019_compute
tb020_compute = tc006_020_compute
tb021_compute = tc006_021_compute
tb022_compute = tc006_022_compute
tb023_compute = tc006_023_compute
tb024_compute = tc006_024_compute
tb025_compute = tc006_025_compute
tb026_compute = tc006_026_compute
tb027_compute = tc006_027_compute
tb028_compute = tc006_028_compute
tb029_compute = tc006_029_compute
tb030_compute = tc006_030_compute
tb031_compute = tc006_031_compute
tb032_compute = tc006_032_compute
tb033_compute = tc006_033_compute
tb034_compute = tc006_034_compute
tb035_compute = tc006_035_compute
tb036_compute = tc006_036_compute
tb037_compute = tc006_037_compute
tb038_compute = tc006_038_compute
tb039_compute = tc006_039_compute
tb040_compute = tc006_040_compute
tb041_compute = tc006_041_compute
tb042_compute = tc006_042_compute
tb043_compute = tc006_043_compute
tb044_compute = tc006_044_compute
tb045_compute = tc006_045_compute
tb046_compute = tc006_046_compute
tb047_compute = tc006_047_compute
tb048_compute = tc006_048_compute
tb049_compute = tc006_049_compute
tb050_compute = tc006_050_compute
tb051_compute = tc006_051_compute
tb052_compute = tc006_052_compute
tb053_compute = tc006_053_compute
tb054_compute = tc006_054_compute
tb055_compute = tc006_055_compute


from .aggregator import (
    FACTOR_ALIASES,
    FACTOR_NAMES,
    FACTORS,
    aggregate_tc006,
    compute as tc006_compute_all,
)

compute_all = tc006_compute_all

__all__ = [
    "FACTOR_ALIASES",
    "FACTOR_NAMES",
    "FACTORS",
    "aggregate_tc006",
    "compute_all",
    "tc006_compute_all",
    "tc006_001_compute",
    "tc006_002_compute",
    "tc006_003_compute",
    "tc006_004_compute",
    "tc006_005_compute",
    "tc006_006_compute",
    "tc006_007_compute",
    "tc006_008_compute",
    "tc006_009_compute",
    "tc006_010_compute",
    "tc006_011_compute",
    "tc006_012_compute",
    "tc006_013_compute",
    "tc006_014_compute",
    "tc006_015_compute",
    "tc006_016_compute",
    "tc006_017_compute",
    "tc006_018_compute",
    "tc006_019_compute",
    "tc006_020_compute",
    "tc006_021_compute",
    "tc006_022_compute",
    "tc006_023_compute",
    "tc006_024_compute",
    "tc006_025_compute",
    "tc006_026_compute",
    "tc006_027_compute",
    "tc006_028_compute",
    "tc006_029_compute",
    "tc006_030_compute",
    "tc006_031_compute",
    "tc006_032_compute",
    "tc006_033_compute",
    "tc006_034_compute",
    "tc006_035_compute",
    "tc006_036_compute",
    "tc006_037_compute",
    "tc006_038_compute",
    "tc006_039_compute",
    "tc006_040_compute",
    "tc006_041_compute",
    "tc006_042_compute",
    "tc006_043_compute",
    "tc006_044_compute",
    "tc006_045_compute",
    "tc006_046_compute",
    "tc006_047_compute",
    "tc006_048_compute",
    "tc006_049_compute",
    "tc006_050_compute",
    "tc006_051_compute",
    "tc006_052_compute",
    "tc006_053_compute",
    "tc006_054_compute",
    "tc006_055_compute",
]
