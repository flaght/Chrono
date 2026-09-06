"""tf002 因子显式导出。"""

from .tf002_001 import compute as tf002_001_compute
from .tf002_002 import compute as tf002_002_compute
from .tf002_003 import compute as tf002_003_compute
from .tf002_004 import compute as tf002_004_compute
from .tf002_005 import compute as tf002_005_compute
from .tf002_006 import compute as tf002_006_compute
from .tf002_007 import compute as tf002_007_compute
from .tf002_008 import compute as tf002_008_compute
from .tf002_009 import compute as tf002_009_compute
from .tf002_010 import compute as tf002_010_compute
from .tf002_011 import compute as tf002_011_compute
from .tf002_012 import compute as tf002_012_compute
from .tf002_013 import compute as tf002_013_compute
from .tf002_014 import compute as tf002_014_compute
from .tf002_015 import compute as tf002_015_compute

# 历史过渡期兼容别名
cr046_compute = tf002_001_compute
cr047_compute = tf002_002_compute
cr048_compute = tf002_003_compute
cr049_compute = tf002_004_compute
cr050_compute = tf002_005_compute
cr052_compute = tf002_006_compute
cr053_compute = tf002_007_compute
cr054_compute = tf002_008_compute
cr055_compute = tf002_009_compute
cr056_compute = tf002_010_compute
cr058_compute = tf002_011_compute
cr060_compute = tf002_012_compute
cr061_compute = tf002_013_compute
cr063_compute = tf002_014_compute
or001_compute = tf002_015_compute

from .aggregator import (
    FACTOR_ALIASES,
    FACTOR_NAMES,
    FACTORS,
    aggregate_tf002,
    compute as tf002_compute_all,
)

compute_all = tf002_compute_all

__all__ = [
    "FACTOR_ALIASES",
    "FACTOR_NAMES",
    "FACTORS",
    "aggregate_tf002",
    "compute_all",
    "tf002_compute_all",
    "tf002_001_compute",
    "tf002_002_compute",
    "tf002_003_compute",
    "tf002_004_compute",
    "tf002_005_compute",
    "tf002_006_compute",
    "tf002_007_compute",
    "tf002_008_compute",
    "tf002_009_compute",
    "tf002_010_compute",
    "tf002_011_compute",
    "tf002_012_compute",
    "tf002_013_compute",
    "tf002_014_compute",
    "tf002_015_compute",
]

