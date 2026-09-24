"""只负责Nautilus内部派生事件的StrategyTemplate桥。"""

from __future__ import annotations

from bomber.model.data import Bar, BarType
from bomber.model.identifiers import InstrumentId
from bomber.trading.config import StrategyConfig
from bomber.trading.strategy import Strategy

from trader.template import StrategyTemplate


class NautilusStrategyEventBridgeConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    data_key: str = "primary_bar"


class NautilusStrategyEventBridge(Strategy):
    """把引擎内部聚合Bar交给已由UnifiedStrategyRunner托管的策略。

    本桥不绑定策略上下文、不管理策略生命周期、不计算目标差额、也不创建
    订单。目标仍沿统一Runner → Portfolio → Planner → Risk → Backend主链执行。
    """

    def __init__(
        self,
        config: NautilusStrategyEventBridgeConfig,
        target_strategy: StrategyTemplate,
    ) -> None:
        super().__init__(config)
        if not config.data_key.strip():
            raise ValueError("data_key不能为空")
        self.target_strategy = target_strategy
        self.bars_forwarded = 0

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self.bars_forwarded += 1
        self.target_strategy._handle_event(self.config.data_key, bar)

    def on_stop(self) -> None:
        self.unsubscribe_bars(self.config.bar_type)
