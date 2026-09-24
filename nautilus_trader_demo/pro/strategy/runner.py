"""把策略与可替换行情、交易适配器装配起来的统一 Runner。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
import threading
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
    TargetUpdateMode,
)
from strategy.portfolio import (
    AccountTargetKey,
    PortfolioCoordinator,
    PositionManager,
    TargetStore,
)
from strategy.ports import ExecutionClientPort, PositionProvider
from strategy.template import StrategyContext, StrategyTemplate


@dataclass(frozen=True)
class _Registration:
    strategy: StrategyTemplate
    data_bindings: tuple[DataBinding, ...]
    execution_routes: dict[str, ExecutionRoute]


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
        *,
        target_store: TargetStore | None = None,
        portfolio_coordinator: PortfolioCoordinator | None = None,
        position_manager: PositionManager | None = None,
    ) -> None:
        self.mode = RuntimeMode(mode)
        self._target_store = target_store or TargetStore()
        self._portfolio_coordinator = portfolio_coordinator or PortfolioCoordinator()
        self._position_manager = position_manager or PositionManager()
        self._position_provider = position_provider or self._position_manager
        self._feeds: dict[str, MarketDataFeed] = {}
        self._clients: dict[str, ExecutionClientPort] = {}
        self._registrations: dict[str, _Registration] = {}
        self._bindings: dict[tuple[str, DataType, InstrumentId], list[tuple[str, DataBinding]]] = defaultdict(list)
        self._attached_feeds: set[str] = set()
        self._market_observers: list[Any] = []
        self._submit_lock = threading.RLock()
        self._started = False

    @property
    def target_store(self) -> TargetStore:
        """返回Runner正在使用的策略目标存储。"""
        return self._target_store

    @property
    def portfolio_coordinator(self) -> PortfolioCoordinator:
        """返回Runner正在使用的账户目标协调器。"""
        return self._portfolio_coordinator

    @property
    def position_manager(self) -> PositionManager:
        """返回Runner正在使用的仓位与在途状态管理器。"""
        return self._position_manager

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

    def add_market_observer(self, observer: Any) -> None:
        """注册行情旁路观察者，例如风控参考价存储。"""
        if self._started:
            raise RuntimeError("Runner 启动后不能再添加行情观察者")
        callback = getattr(observer, "on_market_event", None)
        if not callable(callback):
            raise TypeError("行情观察者必须实现on_market_event(event)")
        if observer not in self._market_observers:
            self._market_observers.append(observer)

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
        if self.mode is not RuntimeMode.HISTORICAL:
            raise RuntimeError("只有 historical 模式可以调用 run_replay")
        if not self._started:
            self.start()
        replay_feeds = [feed for feed in self._feeds.values() if callable(getattr(feed, "replay", None))]
        if len(replay_feeds) != 1:
            raise RuntimeError("replay 模式必须且只能配置一个聚合回放行情源")
        return replay_feeds[0].replay()

    def submit(self, intent: TargetPortfolio) -> None:
        # 不同实时Feed可能从不同网络线程同时触发策略。目标保存、组合净额和向客户端
        # 发布账户快照必须保持同一顺序，不能让较旧快照在较新快照之后到达执行端。
        with self._submit_lock:
            self._submit_locked(intent)

    def _submit_locked(self, intent: TargetPortfolio) -> None:
        registration = self._registrations.get(intent.strategy_id)
        if registration is None:
            raise ValueError(f"未知 strategy_id: {intent.strategy_id}")
        unknown_targets = set(intent.targets) - set(registration.execution_routes)
        if unknown_targets:
            raise ValueError(
                f"策略 {intent.strategy_id} 没有这些target_key的执行路由: "
                f"{sorted(unknown_targets)}",
            )

        # TargetStore先把PATCH物化为完整REPLACE快照，并统一执行revision校验。
        materialized = self._target_store.apply(intent)
        previous = self._portfolio_coordinator.strategy_contribution(intent.strategy_id)
        resolved: dict[AccountTargetKey, Decimal] = {}
        for target_key, quantity in materialized.targets.items():
            route = registration.execution_routes.get(target_key)
            if route is None:
                raise ValueError(
                    f"策略 {intent.strategy_id} 没有 target_key={target_key} 的执行路由",
                )
            account_key = AccountTargetKey(route.client_id, route.instrument_id)
            resolved[account_key] = resolved.get(account_key, Decimal(0)) + quantity

        # materialized已经是完整快照，所以Coordinator也使用REPLACE。它会保留已知腿并
        # 在目标被移除时输出0，确保下游收到明确的清仓目标。
        snapshot = self._portfolio_coordinator.update(
            strategy_id=intent.strategy_id,
            revision=materialized.revision,
            ts_event=materialized.ts_event,
            targets=resolved,
            update_mode=TargetUpdateMode.REPLACE,
        )
        affected_clients = {key.client_id for key in previous} | {
            key.client_id for key in resolved
        }
        for client_id in sorted(affected_clients):
            targets = {
                key.instrument_id: quantity
                for key, quantity in snapshot.targets.items()
                if key.client_id == client_id
            }
            # 对当前策略被REPLACE移除的逻辑腿显式给0，便于审计；账户级targets已是
            # 所有策略在该客户端上的净额快照。
            logical_targets = {
                target_key: materialized.targets.get(target_key, Decimal(0))
                for target_key, route in registration.execution_routes.items()
                if route.client_id == client_id
            }
            client = self._clients[client_id]
            client.submit_targets(
                ExecutionRequest(
                    strategy_id=materialized.strategy_id,
                    revision=materialized.revision,
                    client_id=client_id,
                    ts_event=materialized.ts_event,
                    targets=targets,
                    logical_targets=logical_targets,
                    execution_policy=materialized.execution_policy,
                    deadline_ns=materialized.deadline_ns,
                    metadata={
                        **materialized.metadata,
                        "portfolio_revision": snapshot.revision,
                    },
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
        for observer in tuple(self._market_observers):
            observer.on_market_event(event)
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
