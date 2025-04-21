from enum import Enum
from collections import namedtuple

class STATE(Enum):
    INIT = 0
    LOGIN_ON = 1
    CONTRACT = 2
    MARKET_TICK = 3
    POSITION = 4

class ContractTuple(
        namedtuple("ContractTuple", (
            'getway_name',
            'symbol',
            'exchange',
            'name',
        ))):
    __slots__ = ()

class BarData(object):

    def __init__(self):
        self.vt_symbol = ""
        self.symbol = ""
        self.exchange = ""

        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0

        self.date = ""
        self.time = ""
        self.datetime = None

        self.volume = 0.0
        self.value = 0.0
        self.open_interest = 0.0

class CacheBar(object):

    def __init__(self, symbol):
        self.symbol = symbol
        self.bar = None
        self.minute = 0