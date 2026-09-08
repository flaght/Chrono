"""
数据类型 - 与生产系统 bomber/stgWrapper 完全一致的字段名和访问方式。

这些类型在适配层内部使用，由 Nautilus 事件转换而来。
策略团队看到的是这些类型，不需要知道 Nautilus 的存在。

时间戳说明：
  - timeStamp / timestamp 属性：UTC 纳秒（与生产系统一致）
  - cst_time / cst_hour / cst_minute 属性：北京时间，策略层推荐使用
"""
from datetime import datetime, timezone, timedelta

# 中国标准时间（北京时间，UTC+8）
CST = timezone(timedelta(hours=8))


def _utc_ns_to_cst(ts_ns: int) -> datetime:
    """UTC 纳秒 → 北京时间 datetime"""
    return datetime.fromtimestamp(ts_ns / 1e9, tz=CST)


class BarData:
    """K线数据 - 字段名与生产一致"""
    __slots__ = ("_inst", "_ts", "_timeUnit", "_barCode",
                 "_open", "_high", "_low", "_close", "_volume", "_turnover",
                 "_nautilus_bar")

    def __init__(self, inst="", ts=0, timeUnit=3, barCode=0,
                 openPrice=0.0, highPrice=0.0, lowPrice=0.0, closePrice=0.0,
                 volume=0.0, turnOver=0.0):
        self._inst = inst
        self._ts = ts
        self._timeUnit = timeUnit
        self._barCode = barCode
        self._open = openPrice
        self._high = highPrice
        self._low = lowPrice
        self._close = closePrice
        self._volume = volume
        self._turnover = turnOver
        self._nautilus_bar = None

    def InstStr(self):
        return self._inst

    @property
    def timeStamp(self): return self._ts
    @property
    def timeUnit(self): return self._timeUnit
    @property
    def barCode(self): return self._barCode
    @property
    def openPrice(self): return self._open
    @property
    def highPrice(self): return self._high
    @property
    def lowPrice(self): return self._low
    @property
    def closePrice(self): return self._close
    @property
    def volume(self): return self._volume
    @property
    def turnOver(self): return self._turnover

    # ---- 北京时间便捷方法（策略层推荐用这些，不用关心时区）----

    @property
    def cst_time(self) -> datetime:
        """北京时间 datetime（CST，UTC+8）"""
        return _utc_ns_to_cst(self._ts)

    @property
    def cst_hour(self) -> int:
        """北京时间小时（0-23）"""
        return _utc_ns_to_cst(self._ts).hour

    @property
    def cst_minute(self) -> int:
        """北京时间分钟（0-59）"""
        return _utc_ns_to_cst(self._ts).minute

    def __repr__(self):
        return (f"BarData({self._inst} O={self._open} H={self._high} "
                f"L={self._low} C={self._close} V={self._volume})")


class MarketData:
    """Tick行情数据 - 字段名与生产一致"""
    __slots__ = ("_inst", "_ts", "_exTs", "_level", "_last", "_vol", "_to",
                 "_oi", "_bp", "_bv", "_ap", "_av", "_idx",
                 "_upLmt", "_loLmt", "_preStl", "_openP")

    def __init__(self, inst="", ts=0, exchangeTSMicro=0, Level=1,
                 lastPrice=0.0, volume=0.0, turnover=0.0, openinterest=0.0,
                 bidPrice=None, bidVolume=None, askPrice=None, askVolume=None,
                 Index=0, upLimitPrice=0.0, lowLimitPrice=0.0,
                 preSettlePrice=0.0, openPrice=0.0):
        self._inst = inst
        self._ts = ts
        self._exTs = exchangeTSMicro
        self._level = Level
        self._last = lastPrice
        self._vol = volume
        self._to = turnover
        self._oi = openinterest
        self._bp = bidPrice if bidPrice else [0.0]
        self._bv = bidVolume if bidVolume else [0.0]
        self._ap = askPrice if askPrice else [0.0]
        self._av = askVolume if askVolume else [0.0]
        self._idx = Index
        self._upLmt = upLimitPrice
        self._loLmt = lowLimitPrice
        self._preStl = preSettlePrice
        self._openP = openPrice

    def InstStr(self):
        return self._inst

    @property
    def timeStamp(self): return self._ts
    @property
    def exchangeTSMicro(self): return self._exTs
    @property
    def Level(self): return self._level
    @property
    def lastPrice(self): return self._last
    @property
    def volume(self): return self._vol
    @property
    def turnover(self): return self._to
    @property
    def openinterest(self): return self._oi
    @property
    def bidPrice(self): return self._bp
    @property
    def bidVolume(self): return self._bv
    @property
    def askPrice(self): return self._ap
    @property
    def askVolume(self): return self._av
    @property
    def Index(self): return self._idx
    @property
    def upLimitPrice(self): return self._upLmt
    @property
    def lowLimitPrice(self): return self._loLmt
    @property
    def preSettlePrice(self): return self._preStl
    @property
    def openPrice(self): return self._openP

    # ---- 北京时间便捷方法（策略层推荐用这些，不用关心时区）----

    @property
    def cst_time(self) -> datetime:
        """北京时间 datetime（CST，UTC+8）"""
        return _utc_ns_to_cst(self._ts)

    @property
    def cst_hour(self) -> int:
        """北京时间小时（0-23）"""
        return _utc_ns_to_cst(self._ts).hour

    @property
    def cst_minute(self) -> int:
        """北京时间分钟（0-59）"""
        return _utc_ns_to_cst(self._ts).minute

    def __repr__(self):
        return (f"MarketData({self._inst} last={self._last} "
                f"bid={self._bp[0]} ask={self._ap[0]})")


class Trade:
    """成交数据"""
    __slots__ = ("_tradeId", "_orderId", "_portfolioId", "_price", "_volume",
                 "_ts", "_hedgeFlag", "_direction", "_offset", "_instId")

    def __init__(self, localTradeId=0, localOrderId=0, portfolioId=0,
                 price=0.0, volume=0.0, timeStamp=0, hedgeFlag=0,
                 direction=0, offset=0, instrumentId=""):
        self._tradeId = localTradeId
        self._orderId = localOrderId
        self._portfolioId = portfolioId
        self._price = price
        self._volume = volume
        self._ts = timeStamp
        self._hedgeFlag = hedgeFlag
        self._direction = direction
        self._offset = offset
        self._instId = instrumentId

    @property
    def localTradeId(self): return self._tradeId
    @property
    def localOrderId(self): return self._orderId
    @property
    def portfolioId(self): return self._portfolioId
    @property
    def price(self): return self._price
    @property
    def volume(self): return self._volume
    @property
    def timeStamp(self): return self._ts
    @property
    def hedgeFlag(self): return self._hedgeFlag
    @property
    def direction(self): return self._direction
    @property
    def offset(self): return self._offset
    @property
    def instrumentId(self): return self._instId


class GreeksData:
    """Greeks数据 - 字段名与生产一致"""
    __slots__ = ("_inst", "_delta", "_gamma", "_theta", "_vega", "_rho",
                 "_iv", "_ts")

    def __init__(self, InstStr="", delta=0.0, gamma=0.0, theta=0.0,
                 vega=0.0, rho=0.0, impliedVolatility=0.0, timestamp=0):
        self._inst = InstStr
        self._delta = delta
        self._gamma = gamma
        self._theta = theta
        self._vega = vega
        self._rho = rho
        self._iv = impliedVolatility
        self._ts = timestamp

    @property
    def InstStr(self): return self._inst
    @property
    def delta(self): return self._delta
    @property
    def gamma(self): return self._gamma
    @property
    def theta(self): return self._theta
    @property
    def vega(self): return self._vega
    @property
    def rho(self): return self._rho
    @property
    def impliedVolatility(self): return self._iv
    @property
    def timestamp(self): return self._ts


class IVData:
    """IV数据 - 字段名与生产一致"""
    __slots__ = ("_inst", "_vol", "_price", "_volType", "_ci", "_calcMethod",
                 "_ts", "_fwd", "_isAtm")

    def __init__(self, InstStr="", vol=0.0, price=0.0, volatilityType=1,
                 confidenceInterval=0.0, calculationMethod=1, timestamp=0,
                 forwardUsed=0.0, isAtmForwardOption=False):
        self._inst = InstStr
        self._vol = vol
        self._price = price
        self._volType = volatilityType
        self._ci = confidenceInterval
        self._calcMethod = calculationMethod
        self._ts = timestamp
        self._fwd = forwardUsed
        self._isAtm = isAtmForwardOption

    @property
    def InstStr(self): return self._inst
    @property
    def vol(self): return self._vol
    @property
    def price(self): return self._price
    @property
    def volatilityType(self): return self._volType
    @property
    def confidenceInterval(self): return self._ci
    @property
    def calculationMethod(self): return self._calcMethod
    @property
    def timestamp(self): return self._ts
    @property
    def forwardUsed(self): return self._fwd
    @property
    def isAtmForwardOption(self): return self._isAtm
