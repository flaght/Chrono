"""mf001 期货持仓与博弈形态类特征批次包 (open_interest —— 21 个特征算子)。"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mf001_001 import compute as mf001_001_compute
from .mf001_002 import compute as mf001_002_compute
from .mf001_003 import compute as mf001_003_compute
from .mf001_004 import compute as mf001_004_compute
from .mf001_005 import compute as mf001_005_compute
from .mf001_006 import compute as mf001_006_compute
from .mf001_007 import compute as mf001_007_compute
from .mf001_008 import compute as mf001_008_compute
from .mf001_009 import compute as mf001_009_compute
from .mf001_010 import compute as mf001_010_compute
from .mf001_011 import compute as mf001_011_compute
from .mf001_012 import compute as mf001_012_compute
from .mf001_013 import compute as mf001_013_compute
from .mf001_014 import compute as mf001_014_compute
from .mf001_015 import compute as mf001_015_compute
from .mf001_016 import compute as mf001_016_compute
from .mf001_017 import compute as mf001_017_compute
from .mf001_018 import compute as mf001_018_compute
from .mf001_019 import compute as mf001_019_compute
from .mf001_020 import compute as mf001_020_compute
from .mf001_021 import compute as mf001_021_compute

delta_oi_compute = mf001_001_compute
abs_delta_oi_compute = mf001_002_compute
delta_oi_ratio_compute = mf001_003_compute
abs_delta_oi_ratio_compute = mf001_004_compute
oi_volume_ratio_compute = mf001_005_compute
double_open_vol_compute = mf001_006_compute
double_close_vol_compute = mf001_007_compute
swap_volume_compute = mf001_008_compute
double_open_ratio_compute = mf001_009_compute
double_close_ratio_compute = mf001_010_compute
swap_volume_ratio_compute = mf001_011_compute
bull_active_open_compute = mf001_012_compute
bear_active_open_compute = mf001_013_compute
bull_stop_loss_vol_compute = mf001_014_compute
bear_stop_loss_vol_compute = mf001_015_compute
bull_active_open_ratio_compute = mf001_016_compute
bear_active_open_ratio_compute = mf001_017_compute
bull_stop_loss_ratio_compute = mf001_018_compute
bear_stop_loss_ratio_compute = mf001_019_compute
oi_flow_imbalance_compute = mf001_020_compute
oi_weighted_price_compute = mf001_021_compute

FACTORS = [f"mf001_{i:03d}" for i in range(1, 22)]

FACTOR_ALIASES: dict[str, str] = {
    "mf001_001": "delta_oi",
    "mf001_002": "abs_delta_oi",
    "mf001_003": "delta_oi_ratio",
    "mf001_004": "abs_delta_oi_ratio",
    "mf001_005": "oi_volume_ratio",
    "mf001_006": "double_open_vol",
    "mf001_007": "double_close_vol",
    "mf001_008": "swap_volume",
    "mf001_009": "double_open_ratio",
    "mf001_010": "double_close_ratio",
    "mf001_011": "swap_volume_ratio",
    "mf001_012": "bull_active_open",
    "mf001_013": "bear_active_open",
    "mf001_014": "bull_stop_loss_vol",
    "mf001_015": "bear_stop_loss_vol",
    "mf001_016": "bull_active_open_ratio",
    "mf001_017": "bear_active_open_ratio",
    "mf001_018": "bull_stop_loss_ratio",
    "mf001_019": "bear_stop_loss_ratio",
    "mf001_020": "oi_flow_imbalance",
    "mf001_021": "oi_weighted_price",
}

from .aggregator import aggregate_mf001

compute_all = aggregate_mf001
compute = compute_all

__all__ = [
    "FACTORS",
    "FACTOR_ALIASES",
    "preprocess_ticks",
    "aggregate_mf001",
    "compute_all",
    "compute",
    *(f"mf001_{i:03d}_compute" for i in range(1, 22)),
    *(f"{alias}_compute" for alias in FACTOR_ALIASES.values()),
]
