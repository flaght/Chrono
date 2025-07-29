from enum import Enum


class Offset(Enum):
    NONE = 0
    OPEN = 1  # 开仓
    CLOSE = 2  # 平仓
    CLOSETODAY = 3  # 平今
    CLOSEYESTERDAY = 4  # 平昨
    FORCE_CLOSE = 5  # 强平


class Direction(Enum):
    LONG = 1
    SHORT = -1
    NET = 0


class OrderType(Enum):
    LIMIT = 1  # "限价"
    MARKET = 2  # "市价"
    BEST = 3  # "最优"
    LAST = 4  # "最新"
    AVG = 5  # '均价'
    #STOP = "STOP"
    #FAK = "FAK"
    #FOK = "FOK"
    #RFQ = _("询价")


class OrderStatus(Enum):
    UNKOWN = -1  # 未知
    NOT_TRADED = 0  # 未成交
    ENTRUST_TRADED = 1# 委托成功
    PART_TRADED = 2  # 部分成交
    ALL_TRADED = 3  # 全部成交
    CANCELLED = 4  # 已撤销
    REJECTED = 5  # 拒单


class Interval(Enum):
    """
    Interval of bar data.
    """
    MINUTE = "1m"
    HOUR = "1h"
    DAILY = "d"
    WEEKLY = "w"
    TICK = "tick"


class Exchange(Enum):
    CFFEX = "CFFEX"  # China Financial Futures Exchange
    SHFE = "SHFE"  # Shanghai Futures Exchange
    CZCE = "CZCE"  # Zhengzhou Commodity Exchange
    DCE = "DCE"  # Dalian Commodity Exchange


MAPPING = {'IM': Exchange.CFFEX}
