"""mc004 微观交互与协方差类特征批次包 (corr —— 5 个特征算子)。"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mc004_001 import compute as mc004_001_compute
from .mc004_002 import compute as mc004_002_compute
from .mc004_003 import compute as mc004_003_compute
from .mc004_004 import compute as mc004_004_compute
from .mc004_005 import compute as mc004_005_compute

corr_money_ret_compute = mc004_001_compute
corr_money_spread_compute = mc004_002_compute
corr_ofi_ret_compute = mc004_003_compute
corr_vwap_spread_compute = mc004_004_compute
corr_vol_depth_imb_compute = mc004_005_compute

FACTORS = [f"mc004_{i:03d}" for i in range(1, 6)]

FACTOR_ALIASES: dict[str, str] = {
    "mc004_001": "corr_money_ret",
    "mc004_002": "corr_money_spread",
    "mc004_003": "corr_ofi_ret",
    "mc004_004": "corr_vwap_spread",
    "mc004_005": "corr_vol_depth_imb",
}

from .aggregator import aggregate_mc004

compute_all = aggregate_mc004
compute = compute_all

__all__ = [
    "FACTORS",
    "FACTOR_ALIASES",
    "preprocess_ticks",
    "aggregate_mc004",
    "compute_all",
    "compute",
    *(f"mc004_{i:03d}_compute" for i in range(1, 6)),
    *(f"{alias}_compute" for alias in FACTOR_ALIASES.values()),
]
