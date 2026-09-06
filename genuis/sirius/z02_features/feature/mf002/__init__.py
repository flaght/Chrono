"""mf002 期货涨跌停极限边界与流动性挤压类特征批次包 (limit_bounds —— 4 个特征算子)。"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mf002_001 import compute as mf002_001_compute
from .mf002_002 import compute as mf002_002_compute
from .mf002_003 import compute as mf002_003_compute
from .mf002_004 import compute as mf002_004_compute

dist_upper_limit_compute = mf002_001_compute
dist_lower_limit_compute = mf002_002_compute
limit_bound_asymmetry_compute = mf002_003_compute
near_limit_liquidity_compute = mf002_004_compute

FACTORS = [f"mf002_{i:03d}" for i in range(1, 5)]

FACTOR_ALIASES: dict[str, str] = {
    "mf002_001": "dist_upper_limit",
    "mf002_002": "dist_lower_limit",
    "mf002_003": "limit_bound_asymmetry",
    "mf002_004": "near_limit_liquidity",
}

from .aggregator import aggregate_mf002

compute_all = aggregate_mf002
compute = compute_all

__all__ = [
    "FACTORS",
    "FACTOR_ALIASES",
    "preprocess_ticks",
    "aggregate_mf002",
    "compute_all",
    "compute",
    *(f"mf002_{i:03d}_compute" for i in range(1, 5)),
    *(f"{alias}_compute" for alias in FACTOR_ALIASES.values()),
]
