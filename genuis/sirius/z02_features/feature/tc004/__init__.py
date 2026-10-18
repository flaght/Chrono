"""tc004 因子显式导出。"""

from .tc004_001 import compute as tc004_001_compute
from .tc004_002 import compute as tc004_002_compute
from .tc004_003 import compute as tc004_003_compute
from .tc004_004 import compute as tc004_004_compute
from .tc004_005 import compute as tc004_005_compute
from .tc004_006 import compute as tc004_006_compute
from .tc004_007 import compute as tc004_007_compute
from .tc004_008 import compute as tc004_008_compute
from .tc004_009 import compute as tc004_009_compute
from .tc004_010 import compute as tc004_010_compute
from .tc004_011 import compute as tc004_011_compute
from .tc004_012 import compute as tc004_012_compute
from .tc004_013 import compute as tc004_013_compute
from .tc004_014 import compute as tc004_014_compute
from .tc004_015 import compute as tc004_015_compute
from .tc004_016 import compute as tc004_016_compute
from .tc004_017 import compute as tc004_017_compute
from .tc004_018 import compute as tc004_018_compute
from .tc004_019 import compute as tc004_019_compute
from .tc004_020 import compute as tc004_020_compute
from .tc004_021 import compute as tc004_021_compute

# 历史过渡期兼容别名
dv001_compute = tc004_001_compute
dv002_compute = tc004_002_compute
dv003_compute = tc004_003_compute
dv004_compute = tc004_004_compute
dv005_compute = tc004_005_compute
dv006_compute = tc004_006_compute
dv007_compute = tc004_007_compute
dv008_compute = tc004_008_compute
dv009_compute = tc004_009_compute
dv010_compute = tc004_010_compute
dv011_compute = tc004_011_compute
dv012_compute = tc004_012_compute
tn001_compute = tc004_013_compute
tn002_compute = tc004_014_compute
tn003_compute = tc004_015_compute
tn004_compute = tc004_016_compute
tn005_compute = tc004_017_compute
tn006_compute = tc004_018_compute
tn007_compute = tc004_019_compute
tn008_compute = tc004_020_compute
tn009_compute = tc004_021_compute


from .aggregator import (
    FACTOR_ALIASES,
    FACTOR_NAMES,
    FACTORS,
    aggregate_tc004,
    compute as tc004_compute_all,
)

compute_all = tc004_compute_all

__all__ = [
    "FACTOR_ALIASES",
    "FACTOR_NAMES",
    "FACTORS",
    "aggregate_tc004",
    "compute_all",
    "tc004_compute_all",
    "tc004_001_compute",
    "tc004_002_compute",
    "tc004_003_compute",
    "tc004_004_compute",
    "tc004_005_compute",
    "tc004_006_compute",
    "tc004_007_compute",
    "tc004_008_compute",
    "tc004_009_compute",
    "tc004_010_compute",
    "tc004_011_compute",
    "tc004_012_compute",
    "tc004_013_compute",
    "tc004_014_compute",
    "tc004_015_compute",
    "tc004_016_compute",
    "tc004_017_compute",
    "tc004_018_compute",
    "tc004_019_compute",
    "tc004_020_compute",
    "tc004_021_compute",
]
