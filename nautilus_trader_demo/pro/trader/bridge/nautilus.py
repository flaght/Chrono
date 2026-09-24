"""旧版StrategyTemplate与Bomber/Nautilus StrategyEngine兼容桥。"""

from __future__ import annotations

from decimal import Decimal

from bomber.model.data import Bar
from bomber.model.events import (
    OrderCanceled,
    OrderDenied,
    OrderExpired,
    OrderFilled,
    OrderRejected,
)
from bomber.model.identifiers import InstrumentId
from bomber.model.instruments import Instrument
from bomber.model.data import BarType
from bomber.trading.config import StrategyConfig
from bomber.trading.strategy import Strategy

from trader.contracts import TargetPortfolio
from trader.execution import NautilusExecutionAdapter
from trader.template import StrategyContext, StrategyTemplate


class NautilusStrategyBridgeConfig(StrategyConfig, frozen=True):
    """单标的统一策略的原生引擎托管配置。

    第一阶段明确限制为一个data_key、一个target_key和一个实际合约。后续组合策略
    会在此桥上扩展多数据绑定和多执行路由，不把路由判断放回信号策略。
    """

    instrument_id: InstrumentId
    bar_type: BarType
    data_key: str = "primary_bar"
    target_key: str = "position"
    submit_orders: bool = True
    close_positions_on_stop: bool = False


class _BridgeContext(StrategyContext):
    def __init__(self, bridge: NautilusStrategyBridge) -> None:
        self._bridge = bridge

    def submit(self, intent: TargetPortfolio) -> None:
        self._bridge.accept_target(intent)

    def position(self, target_key: str) -> Decimal:
        return self._bridge.target_position(target_key)


class NautilusStrategyBridge(Strategy):
    """迁移期兼容入口；新装配不应再使用其执行职责。

    新的正式回测主链使用UnifiedStrategyRunner、SimulationExecutionClient和
    NautilusSimExecutionBackend。只有仍直接托管于原生StrategyEngine的旧示例需要
    本类；内部聚合Bar使用不带执行职责的NautilusStrategyEventBridge。
    """

    def __init__(
        self,
        config: NautilusStrategyBridgeConfig,
        target_strategy: StrategyTemplate,
    ) -> None:
        super().__init__(config)
        if not config.data_key.strip() or not config.target_key.strip():
            raise ValueError("data_key和target_key不能为空")
        self.target_strategy = target_strategy
        self.instrument: Instrument | None = None
        self.execution = NautilusExecutionAdapter(
            self,
            strategy_id=target_strategy.strategy_id,
            target_key=config.target_key,
            instrument_id=config.instrument_id,
            submit_orders=config.submit_orders,
        )

    @property
    def target_events(self) -> list[dict[str, str | int]]:
        return self.execution.target_events

    @property
    def order_events(self) -> list[dict[str, str | int]]:
        return self.execution.order_events

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"找不到合约: {self.config.instrument_id}")
            self.stop()
            return
        self.execution.start(self.instrument)
        self.target_strategy._bind(_BridgeContext(self))
        self.target_strategy._start()
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self.target_strategy._handle_event(self.config.data_key, bar)
        self.execution.reconcile(bar.ts_event)

    def accept_target(self, intent: TargetPortfolio) -> None:
        self.execution.submit_target(intent)
        target = intent.targets.get(self.config.target_key, Decimal(0))
        mode = "执行" if self.config.submit_orders else "观察"
        self.log.info(
            f"统一策略目标[{mode}]: strategy={intent.strategy_id} "
            f"revision={intent.revision} target={target}",
        )

    def target_position(self, target_key: str) -> Decimal:
        return self.execution.position(target_key)

    def on_order_filled(self, event: OrderFilled) -> None:
        self.execution.order_filled(event)

    def on_order_rejected(self, event: OrderRejected) -> None:
        self.execution.order_finished(event.client_order_id, event.ts_event, "rejected")

    def on_order_denied(self, event: OrderDenied) -> None:
        self.execution.order_finished(event.client_order_id, event.ts_event, "denied")

    def on_order_canceled(self, event: OrderCanceled) -> None:
        self.execution.order_finished(event.client_order_id, event.ts_event, "canceled")

    def on_order_expired(self, event: OrderExpired) -> None:
        self.execution.order_finished(event.client_order_id, event.ts_event, "expired")

    def on_stop(self) -> None:
        self.unsubscribe_bars(self.config.bar_type)
        self.target_strategy._stop()
        self.target_strategy._unbind()
        if self.config.submit_orders:
            self.cancel_all_orders(self.config.instrument_id)
            if self.config.close_positions_on_stop:
                self.close_all_positions(self.config.instrument_id)
