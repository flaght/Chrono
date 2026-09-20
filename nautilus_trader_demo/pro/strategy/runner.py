"""把策略与可替换行情、交易适配器装配起来的统一 Runner。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from market.basic.base import (
    Bar,
    CustomBar,
    DataType,
    InstrumentId,
    MarketDataFeed,
    QuoteTick,
    TradeTick,
)
from strategy.contracts import (
    DataBinding,
    ExecutionRequest,
    ExecutionRoute,
    RuntimeMode,
    TargetPortfolio,
)
from strategy.ports import ExecutionClientPort, PositionProvider
from strategy.template import StrategyContext, StrategyTemplate


@dataclass(frozen=True)
class _Registration:
    strategy: StrategyTemplate
    data_bindings: tuple[DataBinding, ...]
    execution_routes: dict[str, ExecutionRoute]


class _ZeroPositionProvider:
    def position(self, strategy_id: str, target_key: str) -> Decimal:
        del strategy_id, target_key
        return Decimal(0)


class _RuntimeContext(StrategyContext):
    def __init__(self, runner: UnifiedStrategyRunner, strategy_id: str) -> None:
        self._runner = runner
        self._strategy_id = strategy_id

    def submit(self, intent: TargetPortfolio) -> None:
        self._runner.submit(intent)

    def position(self, target_key: str) -> Decimal:
        return self._runner.position(self._strategy_id, target_key)


class UnifiedStrategyRunner:
    """装配行情源、策略和交易客户端，同时保持策略层无感知。

    同一个策略类可以用于离线回放与实盘。一个 data_key 可以绑定 CTP、
    Binance、DolphinDB 或文件行情；一个 target_key 可以绑定 NT、Bomber、
    vn.py 或后续新增的交易客户端。
    """

    def __init__(
        self,
        mode: RuntimeMode | str,
        position_provider: PositionProvider | None = None,
    ) -> None:
        self.mode = RuntimeMode(mode)
        self._position_provider = position_provider or _ZeroPositionProvider()
        self._feeds: dict[str, MarketDataFeed] = {}
        self._clients: dict[str, ExecutionClientPort] = {}
        self._registrations: dict[str, _Registration] = {}
        self._bindings: dict[tuple[str, DataType, InstrumentId], list[tuple[str, DataBinding]]] = defaultdict(list)
        self._attached_feeds: set[str] = set()
        self._started = False

    def add_data_feed(self, feed_id: str, feed: MarketDataFeed) -> None:
        if self._started:
            raise RuntimeError("Runner 启动后不能再添加行情源")
        if not feed_id.strip() or feed_id in self._feeds:
            raise ValueError(f"feed_id 无效或重复: {feed_id!r}")
        self._feeds[feed_id] = feed

    def add_execution_client(self, client: ExecutionClientPort) -> None:
        if self._started:
            raise RuntimeError("Runner 启动后不能再添加交易客户端")
        if not client.client_id.strip() or client.client_id in self._clients:
            raise ValueError(f"client_id 无效或重复: {client.client_id!r}")
        self._clients[client.client_id] = client

    def add_strategy(
        self,
        strategy: StrategyTemplate,
        *,
        data_bindings: tuple[DataBinding, ...],
        execution_routes: tuple[ExecutionRoute, ...],
    ) -> None:
        if self._started:
            raise RuntimeError("Runner 启动后不能再添加策略")
        if strategy.strategy_id in self._registrations:
            raise ValueError(f"strategy_id 重复: {strategy.strategy_id}")
        data_keys = [binding.data_key for binding in data_bindings]
        if len(data_keys) != len(set(data_keys)):
            raise ValueError(f"策略存在重复 data_key: {strategy.strategy_id}")
        routes = {route.target_key: route for route in execution_routes}
        if len(routes) != len(execution_routes):
            raise ValueError(f"策略存在重复 target_key: {strategy.strategy_id}")
        self._registrations[strategy.strategy_id] = _Registration(
            strategy=strategy,
            data_bindings=tuple(data_bindings),
            execution_routes=routes,
        )

    def start(self) -> None:
        if self._started:
            return
        self._validate_configuration()
        self._bindings.clear()
        for client in self._clients.values():
            client.start()
        for feed_id, feed in self._feeds.items():
            self._attach_feed(feed_id, feed)
        for strategy_id, registration in self._registrations.items():
            registration.strategy._bind(_RuntimeContext(self, strategy_id))
            registration.strategy._start()
        try:
            for feed in self._feeds.values():
                feed.connect()
        except Exception:
            self.stop()
            raise
        self._started = True

    def stop(self) -> None:
        for feed in reversed(tuple(self._feeds.values())):
            feed.disconnect()
        for registration in reversed(tuple(self._registrations.values())):
            registration.strategy._stop()
            registration.strategy._unbind()
        for client in reversed(tuple(self._clients.values())):
            client.stop()
        self._started = False

    def run_replay(self) -> Any:
        if self.mode is not RuntimeMode.REPLAY:
            raise RuntimeError("只有 replay 模式可以调用 run_replay")
        if not self._started:
            self.start()
        replay_feeds = [feed for feed in self._feeds.values() if callable(getattr(feed, "replay", None))]
        if len(replay_feeds) != 1:
            raise RuntimeError("replay 模式必须且只能配置一个聚合回放行情源")
        return replay_feeds[0].replay()

    def submit(self, intent: TargetPortfolio) -> None:
        registration = self._registrations.get(intent.strategy_id)
        if registration is None:
            raise ValueError(f"未知 strategy_id: {intent.strategy_id}")
        grouped_targets: dict[str, dict[InstrumentId, Decimal]] = defaultdict(dict)
        grouped_logical: dict[str, dict[str, Decimal]] = defaultdict(dict)
        for target_key, quantity in intent.targets.items():
            route = registration.execution_routes.get(target_key)
            if route is None:
                raise ValueError(
                    f"策略 {intent.strategy_id} 没有 target_key={target_key} 的执行路由",
                )
            concrete = grouped_targets[route.client_id]
            concrete[route.instrument_id] = concrete.get(route.instrument_id, Decimal(0)) + quantity
            grouped_logical[route.client_id][target_key] = quantity

        for client_id, targets in grouped_targets.items():
            client = self._clients[client_id]
            client.submit_targets(
                ExecutionRequest(
                    strategy_id=intent.strategy_id,
                    revision=intent.revision,
                    client_id=client_id,
                    ts_event=intent.ts_event,
                    targets=targets,
                    logical_targets=grouped_logical[client_id],
                    execution_policy=intent.execution_policy,
                    deadline_ns=intent.deadline_ns,
                    metadata=intent.metadata,
                ),
            )

    def position(self, strategy_id: str, target_key: str) -> Decimal:
        registration = self._registrations.get(strategy_id)
        if registration is None or target_key not in registration.execution_routes:
            raise ValueError(f"未知策略目标: {strategy_id}/{target_key}")
        return self._position_provider.position(strategy_id, target_key)

    def publish(self, feed_id: str, event: Any) -> None:
        """发布标准行情事件，供行情适配器和契约测试使用。"""
        data_type = _event_data_type(event)
        instrument_id = _event_instrument_id(event)
        key = (feed_id, data_type, instrument_id)
        for strategy_id, binding in tuple(self._bindings.get(key, ())):
            if not _bar_spec_matches(binding, event):
                continue
            self._registrations[strategy_id].strategy._handle_event(binding.data_key, event)

    def _validate_configuration(self) -> None:
        if not self._registrations:
            raise ValueError("没有配置策略")
        for strategy_id, registration in self._registrations.items():
            for binding in registration.data_bindings:
                if binding.feed_id not in self._feeds:
                    raise ValueError(
                        f"策略 {strategy_id} 引用了未知行情源 {binding.feed_id}",
                    )
            for route in registration.execution_routes.values():
                if route.client_id not in self._clients:
                    raise ValueError(
                        f"策略 {strategy_id} 引用了未知交易客户端 {route.client_id}",
                    )

    def _attach_feed(self, feed_id: str, feed: MarketDataFeed) -> None:
        if feed_id not in self._attached_feeds:
            feed.register_trade_tick_handler(lambda event, fid=feed_id: self.publish(fid, event))
            feed.register_quote_tick_handler(lambda event, fid=feed_id: self.publish(fid, event))
            feed.register_bar_handler(lambda event, fid=feed_id: self.publish(fid, event))
            feed.register_custom_bar_handler(lambda event, fid=feed_id: self.publish(fid, event))
            self._attached_feeds.add(feed_id)

        for strategy_id, registration in self._registrations.items():
            for binding in registration.data_bindings:
                if binding.feed_id != feed_id:
                    continue
                key = (feed_id, binding.data_type, binding.instrument_id)
                self._bindings[key].append((strategy_id, binding))
                feed.subscribe(
                    binding.instrument_id,
                    binding.data_type,
                    bar_spec=binding.bar_spec,
                )


def _event_data_type(event: Any) -> DataType:
    if isinstance(event, TradeTick):
        return DataType.TRADE_TICK
    if isinstance(event, QuoteTick):
        return DataType.QUOTE_TICK
    if isinstance(event, CustomBar):
        return DataType.CUSTOM_BAR
    if isinstance(event, Bar):
        return DataType.BAR
    raise TypeError(f"不支持的策略行情事件: {type(event).__name__}")


def _event_instrument_id(event: Any) -> InstrumentId:
    instrument_id = getattr(event, "instrument_id", None)
    if instrument_id is not None:
        return instrument_id
    base_bar = getattr(event, "bar", event)
    bar_type = getattr(base_bar, "bar_type", None)
    if bar_type is None:
        raise TypeError(f"行情事件缺少 instrument_id: {type(event).__name__}")
    return bar_type.instrument_id


def _bar_spec_matches(binding: DataBinding, event: Any) -> bool:
    if binding.bar_spec is None:
        return True
    base_bar = getattr(event, "bar", event)
    bar_type = getattr(base_bar, "bar_type", None)
    return bar_type is not None and f"-{binding.bar_spec}-" in str(bar_type)
