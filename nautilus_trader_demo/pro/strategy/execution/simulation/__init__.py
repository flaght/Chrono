"""通用模拟执行后端及其交易场所规则配置。"""

from strategy.execution.simulation.backend import NautilusSimExecutionBackend
from strategy.execution.simulation.binance import BinanceUsdtFuturesProfile
from strategy.execution.simulation.ctp import CtpFuturesBasicProfile
from strategy.execution.simulation.profile import GenericVenueProfile

__all__ = [
    "BinanceUsdtFuturesProfile",
    "CtpFuturesBasicProfile",
    "GenericVenueProfile",
    "NautilusSimExecutionBackend",
]
