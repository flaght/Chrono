"""可复用且不依赖具体行情源和交易客户端的策略模板。"""

from __future__ import annotations

import threading
from abc import ABC
from decimal import Decimal
from typing import Any, Mapping, Protocol

from market.basic.base import Bar, CustomBar, QuoteTick, TradeTick
from strategy.contracts import TargetPortfolio, TargetUpdateMode


class StrategyContext(Protocol):
    def submit(self, intent: TargetPortfolio) -> None: ...

    def position(self, target_key: str) -> Decimal: ...


class StrategyTemplate(ABC):
    """单标的、价差和组合策略共用的基础模板。

    策略只接收逻辑 ``data_key``，只输出逻辑 ``target_key``，不会直接获得
    行情源或交易客户端实例。
    """

    def __init__(self, strategy_id: str) -> None:
        if not strategy_id.strip():
            raise ValueError("strategy_id 不能为空")
        self.strategy_id = strategy_id
        self._context: StrategyContext | None = None
        self._revision = 0
        self._revision_lock = threading.Lock()
        self._started = False

    @property
    def is_started(self) -> bool:
        return self._started

    def _bind(self, context: StrategyContext) -> None:
        if self._context is not None:
            raise RuntimeError(f"策略已经绑定运行上下文: {self.strategy_id}")
        self._context = context

    def _unbind(self) -> None:
        if self._started:
            raise RuntimeError(f"不能解绑正在运行的策略: {self.strategy_id}")
        self._context = None

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        self.on_start()

    def _stop(self) -> None:
        if not self._started:
            return
        try:
            self.on_stop()
        finally:
            self._started = False

    def _handle_event(self, data_key: str, event: Any) -> None:
        if not self._started:
            return
        self.on_data(data_key, event)
        if isinstance(event, TradeTick):
            self.on_trade_tick(data_key, event)
        elif isinstance(event, QuoteTick):
            self.on_quote_tick(data_key, event)
        elif isinstance(event, CustomBar):
            self.on_custom_bar(data_key, event)
        elif isinstance(event, Bar):
            self.on_bar(data_key, event)

    def set_target(
        self,
        target_key: str,
        quantity: Decimal | int | float | str,
        ts_event: int,
        *,
        update_mode: TargetUpdateMode | str = TargetUpdateMode.REPLACE,
        execution_policy: str = "DIRECT",
        deadline_ns: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        return self.set_targets(
            {target_key: quantity},
            ts_event,
            update_mode=update_mode,
            execution_policy=execution_policy,
            deadline_ns=deadline_ns,
            metadata=metadata,
        )

    def set_targets(
        self,
        targets: Mapping[str, Decimal | int | float | str],
        ts_event: int,
        *,
        update_mode: TargetUpdateMode | str = TargetUpdateMode.REPLACE,
        execution_policy: str = "DIRECT",
        deadline_ns: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        context = self._require_context()
        with self._revision_lock:
            self._revision += 1
            revision = self._revision
        context.submit(
            TargetPortfolio(
                strategy_id=self.strategy_id,
                revision=revision,
                ts_event=ts_event,
                targets=targets,
                update_mode=TargetUpdateMode(update_mode),
                execution_policy=execution_policy,
                deadline_ns=deadline_ns,
                metadata=metadata or {},
            ),
        )
        return revision

    def position(self, target_key: str) -> Decimal:
        return self._require_context().position(target_key)

    def _require_context(self) -> StrategyContext:
        if self._context is None:
            raise RuntimeError(f"策略尚未绑定运行上下文: {self.strategy_id}")
        return self._context

    def on_start(self) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def on_data(self, data_key: str, event: Any) -> None:
        pass

    def on_trade_tick(self, data_key: str, tick: TradeTick) -> None:
        pass

    def on_quote_tick(self, data_key: str, tick: QuoteTick) -> None:
        pass

    def on_bar(self, data_key: str, bar: Bar) -> None:
        pass

    def on_custom_bar(self, data_key: str, bar: CustomBar) -> None:
        pass
