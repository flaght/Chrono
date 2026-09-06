"""mc001 微观结构资金流向与大单特征批次包 (money_flow —— 25 个特征算子)。

本批次包含 25 个面向 CTP Level-1 Tick 数据降频至 1 分钟的微观资金流因子：
1. 成交笔数类 (6个): tick_in, tick_out, net_tick_in, tick_in_pct, tick_out_pct, net_tick_in_pct
2. 成交手数类 (6个): volume_in, volume_out, net_volume_in, volume_in_pct, volume_out_pct, net_volume_in_pct
3. 成交金额类 (4个): money_in, money_out, net_money_in, net_money_in_pct
4. 聪明钱大单类 (9个): smart_volume_in, smart_volume_out, smart_volume_in_pct, smart_volume_out_pct,
                     smart_money_in, smart_money_out, smart_money_in_pct, smart_money_out_pct, smart_net_vol_pct

使用模式:
- 单因子独立提取: 导入对应因子的 compute 函数，如 `from feature.mc001 import mc001_001_compute` 或 `tick_in_compute`。
- 全量批次高效聚合: 调用 `compute_all(df_lazy)` 或 `aggregate_mc001(df_lazy, use_aliases=True)`，单次扫描 Tick 数据并行完成全部 25 个特征计算。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

from .mc001_001 import compute as mc001_001_compute
from .mc001_002 import compute as mc001_002_compute
from .mc001_003 import compute as mc001_003_compute
from .mc001_004 import compute as mc001_004_compute
from .mc001_005 import compute as mc001_005_compute
from .mc001_006 import compute as mc001_006_compute
from .mc001_007 import compute as mc001_007_compute
from .mc001_008 import compute as mc001_008_compute
from .mc001_009 import compute as mc001_009_compute
from .mc001_010 import compute as mc001_010_compute
from .mc001_011 import compute as mc001_011_compute
from .mc001_012 import compute as mc001_012_compute
from .mc001_013 import compute as mc001_013_compute
from .mc001_014 import compute as mc001_014_compute
from .mc001_015 import compute as mc001_015_compute
from .mc001_016 import compute as mc001_016_compute
from .mc001_017 import compute as mc001_017_compute
from .mc001_018 import compute as mc001_018_compute
from .mc001_019 import compute as mc001_019_compute
from .mc001_020 import compute as mc001_020_compute
from .mc001_021 import compute as mc001_021_compute
from .mc001_022 import compute as mc001_022_compute
from .mc001_023 import compute as mc001_023_compute
from .mc001_024 import compute as mc001_024_compute
from .mc001_025 import compute as mc001_025_compute

# 历史向后兼容别名函数
tick_in_compute = mc001_001_compute
tick_out_compute = mc001_002_compute
net_tick_in_compute = mc001_003_compute
tick_in_pct_compute = mc001_004_compute
tick_out_pct_compute = mc001_005_compute
net_tick_in_pct_compute = mc001_006_compute
volume_in_compute = mc001_007_compute
volume_out_compute = mc001_008_compute
net_volume_in_compute = mc001_009_compute
volume_in_pct_compute = mc001_010_compute
volume_out_pct_compute = mc001_011_compute
net_volume_in_pct_compute = mc001_012_compute
money_in_compute = mc001_013_compute
money_out_compute = mc001_014_compute
net_money_in_compute = mc001_015_compute
net_money_in_pct_compute = mc001_016_compute
smart_volume_in_compute = mc001_017_compute
smart_volume_out_compute = mc001_018_compute
smart_volume_in_pct_compute = mc001_019_compute
smart_volume_out_pct_compute = mc001_020_compute
smart_money_in_compute = mc001_021_compute
smart_money_out_compute = mc001_022_compute
smart_money_in_pct_compute = mc001_023_compute
smart_money_out_pct_compute = mc001_024_compute
smart_net_vol_pct_compute = mc001_025_compute

FACTORS = [
    f"mc001_{i:03d}" for i in range(1, 26)
]

FACTOR_ALIASES: dict[str, str] = {
    "mc001_001": "tick_in",
    "mc001_002": "tick_out",
    "mc001_003": "net_tick_in",
    "mc001_004": "tick_in_pct",
    "mc001_005": "tick_out_pct",
    "mc001_006": "net_tick_in_pct",
    "mc001_007": "volume_in",
    "mc001_008": "volume_out",
    "mc001_009": "net_volume_in",
    "mc001_010": "volume_in_pct",
    "mc001_011": "volume_out_pct",
    "mc001_012": "net_volume_in_pct",
    "mc001_013": "money_in",
    "mc001_014": "money_out",
    "mc001_015": "net_money_in",
    "mc001_016": "net_money_in_pct",
    "mc001_017": "smart_volume_in",
    "mc001_018": "smart_volume_out",
    "mc001_019": "smart_volume_in_pct",
    "mc001_020": "smart_volume_out_pct",
    "mc001_021": "smart_money_in",
    "mc001_022": "smart_money_out",
    "mc001_023": "smart_money_in_pct",
    "mc001_024": "smart_money_out_pct",
    "mc001_025": "smart_net_vol_pct",
}


from .aggregator import aggregate_mc001

compute_all = aggregate_mc001
compute = compute_all


__all__ = [
    "FACTORS",
    "FACTOR_ALIASES",
    "preprocess_ticks",
    "compute_all",
    "compute",
    *(f"mc001_{i:03d}_compute" for i in range(1, 26)),
    *(f"{alias}_compute" for alias in FACTOR_ALIASES.values()),
]
