"""tc003 因子显式导出。"""

from .tc003_001 import compute as tc003_001_compute
from .tc003_002 import compute as tc003_002_compute
from .tc003_003 import compute as tc003_003_compute
from .tc003_004 import compute as tc003_004_compute
from .tc003_005 import compute as tc003_005_compute
from .tc003_006 import compute as tc003_006_compute
from .tc003_007 import compute as tc003_007_compute
from .tc003_008 import compute as tc003_008_compute
from .tc003_009 import compute as tc003_009_compute
from .tc003_010 import compute as tc003_010_compute
from .tc003_011 import compute as tc003_011_compute
from .tc003_012 import compute as tc003_012_compute
from .tc003_013 import compute as tc003_013_compute
from .tc003_014 import compute as tc003_014_compute
from .tc003_015 import compute as tc003_015_compute

# 历史过渡期兼容别名
fz002_compute = tc003_001_compute
gd002_compute = tc003_002_compute
gd003_compute = tc003_003_compute
ha004_compute = tc003_004_compute
ha005_compute = tc003_005_compute
tf001_compute = tc003_006_compute
tf002_compute = tc003_007_compute
tf003_compute = tc003_008_compute
tf004_compute = tc003_009_compute
tf005_compute = tc003_010_compute
tf006_compute = tc003_011_compute
tf008_compute = tc003_012_compute
tf019_compute = tc003_013_compute
tf020_compute = tc003_014_compute
tf022_compute = tc003_015_compute


from .aggregator import (
    FACTOR_ALIASES,
    FACTOR_NAMES,
    FACTORS,
    aggregate_tc003,
    compute as tc003_compute_all,
)

compute_all = tc003_compute_all

__all__ = [
    "FACTOR_ALIASES",
    "FACTOR_NAMES",
    "FACTORS",
    "aggregate_tc003",
    "compute_all",
    "tc003_compute_all",
    "tc003_001_compute",
    "tc003_002_compute",
    "tc003_003_compute",
    "tc003_004_compute",
    "tc003_005_compute",
    "tc003_006_compute",
    "tc003_007_compute",
    "tc003_008_compute",
    "tc003_009_compute",
    "tc003_010_compute",
    "tc003_011_compute",
    "tc003_012_compute",
    "tc003_013_compute",
    "tc003_014_compute",
    "tc003_015_compute",
]
