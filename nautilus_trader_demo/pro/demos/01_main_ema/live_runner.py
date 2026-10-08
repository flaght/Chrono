"""固定本次主力的在线模块装配；主力改变时关闭交易闸门。"""

from __future__ import annotations

from bomber.framework.datahub.sector_roles import SectorDataUnavailable
from bomber.framework.market.basic.base import Bar, DataType, InstrumentId
from bomber.framework.trader import DataBinding, ExecutionRoute, RuntimeMode, UnifiedStrategyRunner

from .strategy import MainEmaStrategy


class MainEmaSimnowRunner(UnifiedStrategyRunner):
    """本根Bar决策完成后再推进保留目标，避免用旧方向抢先开仓。

    LIVE 动态换约仍被公共框架禁用。本类只装配一个固定主力路由，
    主力变更后撤销下单授权、请求撤单并标记需恢复，禁止继续交易。
    不连接柜台、不授权报单，不承担资料加载、预热或跨进程恢复。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode is not RuntimeMode.LIVE:
            raise ValueError("MainEmaSimnowRunner 仅用于 LIVE 固定主力装配")
        self._last_live_bar_ns = {}
        self._main_binding: tuple[MainEmaStrategy, str, InstrumentId, str] | None = None
        self._main_change_reason: str | None = None

    def add_main_strategy(
        self, strategy: MainEmaStrategy, *, feed_id: str,
        instrument_id: InstrumentId, client_id: str, bar_spec: str = "1-MINUTE",
    ) -> None:
        if self._main_binding is not None:
            raise ValueError("固定主力在线装配仅支持一个策略")
        if str(instrument_id.venue).upper() != strategy.config.venue:
            raise ValueError("固定主力合约交易所与策略配置不一致")
        self.add_strategy(
            strategy,
            data_bindings=(DataBinding(
                str(instrument_id), feed_id, instrument_id, DataType.BAR, bar_spec,
            ),),
            execution_routes=(ExecutionRoute(
                strategy.config.target_key, client_id, instrument_id,
            ),),
        )
        self._main_binding = (strategy, feed_id, instrument_id, client_id)

    def _validate_configuration(self) -> None:
        super()._validate_configuration()
        if self._main_binding is None or len(self._registrations) != 1:
            raise ValueError("请用 add_main_strategy 装配单策略固定主力路由")

    def _close_main_gate(self, reason: str) -> None:
        self._main_change_reason = reason
        strategy, _, _, client_id = self._main_binding
        self.position_manager.mark_recovery_required(client_id)
        client = self._clients[client_id]
        disarm = getattr(client, "disarm", None)
        if callable(disarm):
            disarm(reason)
        client.cancel_strategy(strategy.strategy_id)
        raise RuntimeError(reason)

    def publish(self, feed_id, event):
        if not isinstance(event, Bar) or self._main_binding is None:
            return super().publish(feed_id, event)
        with self._submit_lock:
            strategy, main_feed_id, fixed_id, _ = self._main_binding
            if feed_id != main_feed_id:
                return super().publish(feed_id, event)
            if self._main_change_reason is not None:
                raise RuntimeError(self._main_change_reason)
            key = (feed_id, event.bar_type.instrument_id)
            if event.ts_event <= self._last_live_bar_ns.get(key, -1):
                return
            self._last_live_bar_ns[key] = event.ts_event
            reference_ready = False
            try:
                assignment = strategy.data_hub.snapshot(event.ts_event)
                main = assignment.instrument(strategy.config.product, "main").lower()
                if main != str(fixed_id.symbol).lower():
                    self._close_main_gate(
                        f"当前主力已由 {fixed_id.symbol} 变为 {main}；固定路由已闭闸，"
                        "须核对仓位与活动订单并重新装配，LIVE 动态换约尚未启用",
                    )
                assignment.factor(strategy.config.product, "main")
                reference_ready = True
            except SectorDataUnavailable:
                # 策略仍可计数资料不可用事件，但不得用保留目标继续发单。
                pass
            super().publish(feed_id, event)
            if self._started and reference_ready and strategy.last_processed_ns == event.ts_event:
                self.continue_execution_target(
                    strategy.strategy_id, strategy.config.target_key, event.ts_event,
                    trigger_instrument_id=event.bar_type.instrument_id,
                )
