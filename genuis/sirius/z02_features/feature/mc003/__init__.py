"""mc003 微观结构一档订单流动力学特征批次包 (order_flow —— 11 个特征算子)。"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mc003_001 import compute as mc003_001_compute
from .mc003_002 import compute as mc003_002_compute
from .mc003_003 import compute as mc003_003_compute
from .mc003_004 import compute as mc003_004_compute
from .mc003_005 import compute as mc003_005_compute
from .mc003_006 import compute as mc003_006_compute
from .mc003_007 import compute as mc003_007_compute
from .mc003_008 import compute as mc003_008_compute
from .mc003_009 import compute as mc003_009_compute
from .mc003_010 import compute as mc003_010_compute
from .mc003_011 import compute as mc003_011_compute

ofi_sum_compute = mc003_001_compute
ofi_mean_compute = mc003_002_compute
ofi_std_compute = mc003_003_compute
ofi_normalized_compute = mc003_004_compute
voi_1_compute = mc003_005_compute
voi_normalized_compute = mc003_006_compute
bid_depletion_rate_compute = mc003_007_compute
ask_depletion_rate_compute = mc003_008_compute
bid_depletion_ratio_compute = mc003_009_compute
ask_depletion_ratio_compute = mc003_010_compute
depletion_imbalance_compute = mc003_011_compute

FACTORS = [f"mc003_{i:03d}" for i in range(1, 12)]

FACTOR_ALIASES: dict[str, str] = {
    "mc003_001": "ofi_sum",
    "mc003_002": "ofi_mean",
    "mc003_003": "ofi_std",
    "mc003_004": "ofi_normalized",
    "mc003_005": "voi_1",
    "mc003_006": "voi_normalized",
    "mc003_007": "bid_depletion_rate",
    "mc003_008": "ask_depletion_rate",
    "mc003_009": "bid_depletion_ratio",
    "mc003_010": "ask_depletion_ratio",
    "mc003_011": "depletion_imbalance",
}

from .aggregator import aggregate_mc003

compute_all = aggregate_mc003
compute = compute_all

__all__ = [
    "FACTORS",
    "FACTOR_ALIASES",
    "preprocess_ticks",
    "aggregate_mc003",
    "compute_all",
    "compute",
    *(f"mc003_{i:03d}_compute" for i in range(1, 12)),
    *(f"{alias}_compute" for alias in FACTOR_ALIASES.values()),
]
