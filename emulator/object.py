import time
from dataclasses import dataclass, field
from datetime import datetime

from constant import Exchange, OrderType, Direction, Offset, OrderStatus, Interval

ACTIVE_STATUSES = set([OrderStatus.NOT_TRADED, OrderStatus.PART_TRADED])

GLOBAL_ORDER_ID = 100
GLOBAL_TRADE_ID = 100

@dataclass
class BaseData:
    pass


@dataclass
class TradeData(BaseData):
    symbol: str
    orderid: str
    tradeid: str
    strategy_id: str

    tradehd: str #  # 对应持仓单 (Holding ID) 
    direction: Direction = None
    offset: Offset = Offset.NONE
    price1: float = 0  ## 当时成交价
    price2: float = 0  ## 经过结算价刷新价格
    volume: float = 0
    create_time: datetime = None

    @classmethod
    def create_trade_id(cls):
        global GLOBAL_TRADE_ID
        GLOBAL_TRADE_ID += 1
        return GLOBAL_TRADE_ID


@dataclass
class OrderData(BaseData):
    symbol: str
    order_id: str
    strategy_id: str

    tradeid: str ## 成交持仓ID
    is_clear: int # 撮合不成功，进行撮合
    order_type: OrderType = OrderType.LIMIT  # 订单类型 ## 1 限价 2.市场价
    direction: Direction = None
    offset: Offset = Offset.NONE
    status: OrderStatus = OrderStatus.UNKOWN
    price: float = 0
    volume: float = 0  # 下单量
    #traded: float = 0  # 具体成交量
    create_time: datetime = None

    def is_active(self) -> bool:
        """
        Check if the order is active.
        """
        return self.status in ACTIVE_STATUSES
    
    @classmethod
    def create_order_id(cls):
        global GLOBAL_ORDER_ID
        GLOBAL_ORDER_ID += 1
        return GLOBAL_ORDER_ID


@dataclass
class BarData(BaseData):
    code: str
    symbol: str
    exchange: Exchange
    create_time: datetime

    interval: Interval = None
    volume: float = 0
    turnover: float = 0
    open_interest: float = 0
    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    close_price: float = 0
    settle_price: float = 0

@dataclass
class TickData(BaseData):
    """
    Tick data contains information about:
        * last trade in market
        * orderbook snapshot
        * intraday market statistics.
    """

    code: str
    symbol: str
    exchange: Exchange
    create_time: datetime

    name: str = ""
    volume: float = 0
    turnover: float = 0
    open_interest: float = 0
    last_price: float = 0
    last_volume: float = 0
    limit_up: float = 0
    limit_down: float = 0

    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    pre_close: float = 0

    bid_price_1: float = 0
    bid_price_2: float = 0
    bid_price_3: float = 0
    bid_price_4: float = 0
    bid_price_5: float = 0

    ask_price_1: float = 0
    ask_price_2: float = 0
    ask_price_3: float = 0
    ask_price_4: float = 0
    ask_price_5: float = 0

    bid_volume_1: float = 0
    bid_volume_2: float = 0
    bid_volume_3: float = 0
    bid_volume_4: float = 0
    bid_volume_5: float = 0

    ask_volume_1: float = 0
    ask_volume_2: float = 0
    ask_volume_3: float = 0
    ask_volume_4: float = 0
    ask_volume_5: float = 0

    localtime: datetime = None

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
