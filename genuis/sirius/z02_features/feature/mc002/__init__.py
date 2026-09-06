"""mc002 微观结构买卖盘口与微观结构类特征批次包 (imbalance —— 10 个特征算子)。"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mc002_001 import compute as mc002_001_compute
from .mc002_002 import compute as mc002_002_compute
from .mc002_003 import compute as mc002_003_compute
from .mc002_004 import compute as mc002_004_compute
from .mc002_005 import compute as mc002_005_compute
from .mc002_006 import compute as mc002_006_compute
from .mc002_007 import compute as mc002_007_compute
from .mc002_008 import compute as mc002_008_compute
from .mc002_009 import compute as mc002_009_compute
from .mc002_010 import compute as mc002_010_compute

bid_ask_spread_compute = mc002_001_compute
spread_std_compute = mc002_002_compute
relative_spread_compute = mc002_003_compute
depth_imbalance_1_compute = mc002_004_compute
depth_imbalance_std_compute = mc002_005_compute
depth_imbalance_last_compute = mc002_006_compute
micro_price_bias_compute = mc002_007_compute
realized_volatility_compute = mc002_008_compute
realized_bipower_var_compute = mc002_009_compute
jump_ratio_compute = mc002_010_compute

FACTORS = [f"mc002_{i:03d}" for i in range(1, 11)]

FACTOR_ALIASES: dict[str, str] = {
    "mc002_001": "bid_ask_spread",
    "mc002_002": "spread_std",
    "mc002_003": "relative_spread",
    "mc002_004": "depth_imbalance_1",
    "mc002_005": "depth_imbalance_std",
    "mc002_006": "depth_imbalance_last",
    "mc002_007": "micro_price_bias",
    "mc002_008": "realized_volatility",
    "mc002_009": "realized_bipower_var",
    "mc002_010": "jump_ratio",
}

from .aggregator import aggregate_mc002

compute_all = aggregate_mc002
compute = compute_all

__all__ = [
    "FACTORS",
    "FACTOR_ALIASES",
    "preprocess_ticks",
    "aggregate_mc002",
    "compute_all",
    "compute",
    *(f"mc002_{i:03d}_compute" for i in range(1, 11)),
    *(f"{alias}_compute" for alias in FACTOR_ALIASES.values()),
]
