"""mf003 期货日内均价线与基准锚点类特征批次包 (average_price_anchor —— 4 个特征算子)。"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mf003_001 import compute as mf003_001_compute
from .mf003_002 import compute as mf003_002_compute
from .mf003_003 import compute as mf003_003_compute
from .mf003_004 import compute as mf003_004_compute

price_to_avg_dev_compute = mf003_001_compute
vwap_to_avg_bias_compute = mf003_002_compute
avg_price_slope_compute = mf003_003_compute
price_above_avg_time_compute = mf003_004_compute

FACTORS = [f"mf003_{i:03d}" for i in range(1, 5)]

FACTOR_ALIASES: dict[str, str] = {
    "mf003_001": "price_to_avg_dev",
    "mf003_002": "vwap_to_avg_bias",
    "mf003_003": "avg_price_slope",
    "mf003_004": "price_above_avg_time",
}

from .aggregator import aggregate_mf003

compute_all = aggregate_mf003
compute = compute_all

__all__ = [
    "FACTORS",
    "FACTOR_ALIASES",
    "preprocess_ticks",
    "aggregate_mf003",
    "compute_all",
    "compute",
    *(f"mf003_{i:03d}_compute" for i in range(1, 5)),
    *(f"{alias}_compute" for alias in FACTOR_ALIASES.values()),
]
