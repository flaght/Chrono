"""
枚举类型 - 与生产系统完全一致
"""
from enum import IntEnum


class bar_time_unit_t(IntEnum):
    S1 = 0; S5 = 1; S15 = 2
    M1 = 3; M5 = 4; M15 = 5
    H1 = 6; D1 = 7; W1 = 8; TMax = 9


class HedgeFlag_type_t(IntEnum):
    Speculation = 0; Arbitrage = 1; Hedge = 2


class OrderMsg_offset_t(IntEnum):
    Open = 0; Close = 1; CloseToday = 2; CloseYd = 3


class OrderMsg_dir_t(IntEnum):
    Buy = 0; Sell = 1


PERIOD_MAP = {
    "S1": ("SECOND", 1), "S5": ("SECOND", 5), "S15": ("SECOND", 15),
    "M1": ("MINUTE", 1), "M5": ("MINUTE", 5), "M15": ("MINUTE", 15),
    "H1": ("HOUR", 1), "D1": ("DAY", 1), "W1": ("WEEK", 1),
}
