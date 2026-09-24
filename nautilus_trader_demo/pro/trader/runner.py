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
from trader.contracts import (
    DataBinding,
    ExecutionRequest,
    ExecutionRoute,
    RuntimeMode,
    TargetPortfolio,
    TargetUpdateMode,
)
from trader.portfolio import (
    AccountTargetKey,
    PortfolioCoordinator,
    PositionManager,
    TargetStore,
)
from trader.market_health import MarketHealthGate, RecoveryConfirmation
from trader.execution.events import FillEvent, OrderUpdateEvent
from trader.dynamic_routes import DynamicExecutionRoute, RollPhase, SafeRollCoordinator
from trader.ports import ExecutionClientPort, PositionProvider
from trader.template import StrategyContext, StrategyTemplate


@dataclass(frozen=True)
class _Registration:
    strategy: StrategyTemplate
    data_bindings: tuple[DataBinding, ...]
    execution_routes: dict[str, ExecutionRoute | DynamicExecutionRoute]
    time_feed_ids: tuple[str, ...]


class _RuntimeContext(StrategyContext):
    def __init__(self, runner: UnifiedStrategyRunner, strategy_id: str) -> None:
        self._runner = runner
        self._strategy_id = strategy_id

    def submit(self, intent: TargetPortfolio) -> None:
        self._runner.submit(intent)

    def position(self, target_key: str) -> Decimal:
        return self._runner.position(self._strategy_id, target_key)

    def account_position(self, target_key: str) -> Decimal:
        return self._runner.account_position(self._strategy_id, target_key)

    def working_quantity(self, target_key: str) -> Decimal:
        return self._runner.working_quantity(self._strategy_id, target_key)


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
        market_health_gate: MarketHealthGate | None = None,
        roll_coordinator: SafeRollCoordinator | None = None,
    ) -> None:
        self.mode = RuntimeMode(mode)
        self._target_store = target_store or TargetStore()
        self._portfolio_coordinator = portfolio_coordinator or PortfolioCoordinator()
        self._position_manager = position_manager or PositionManager()
        self._position_provider = position_provider or self._position_manager
        self._market_health_gate = market_health_gate or MarketHealthGate()
        self._roll_coordinator = roll_coordinator or SafeRollCoordinator()
        self._feeds: dict[str, MarketDataFeed] = {}
        self._clients: dict[str, ExecutionClientPort] = {}
        self._registrations: dict[str, _Registration] = {}
        self._has_dynamic_routes = False
        self._bindings: dict[tuple[str, DataType, InstrumentId], list[tuple[str, DataBinding]]] = defaultdict(list)
        self._attached_feeds: set[str] = set()
        self._attached_health_feeds: set[str] = set()
        self._market_observers: list[Any] = []
        self._latest_market_event_ns: dict[InstrumentId, int] = {}
        self._publishing_strategy_event = False
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

    @property
    def market_health_gate(self) -> MarketHealthGate:
        return self._market_health_gate

    @property
    def roll_coordinator(self) -> SafeRollCoordinator:
        return self._roll_coordinator

    def add_data_feed(self, feed_id: str, feed: MarketDataFeed) -> None:
        if self._started:
            raise RuntimeError("Runner 启动后不能再添加行情源")
        if not feed_id.strip() or feed_id in self._feeds:
            raise ValueError(f"feed_id 无效或重复: {feed_id!r}")
        self._feeds[feed_id] = feed
        snapshot = getattr(feed, "health_snapshot", None)
        if snapshot is not None:
            self._market_health_gate.register_feed(feed_id, snapshot)

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
        execution_routes: tuple[ExecutionRoute | DynamicExecutionRoute, ...],
        time_feed_ids: tuple[str, ...] = (),
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
        if len(set(time_feed_ids)) != len(time_feed_ids) or any(not item.strip() for item in time_feed_ids):
            raise ValueError(f"策略存在重复或空的时钟 feed_id: {strategy.strategy_id}")
        dynamic = [route for route in execution_routes if isinstance(route, DynamicExecutionRoute)]
        if dynamic:
            if self.mode is RuntimeMode.LIVE:
                raise ValueError("动态换月尚未接入生产级回报恢复与持久化，当前仅允许HISTORICAL")
            # 首阶段安全边界：单逻辑目标占用专属账户；不把多腿换月误认为原子执行。
            if len(dynamic) != 1 or len(routes) != 1:
                raise ValueError("动态路由阶段仅支持单逻辑目标；多腿期权需独立协调策略")
            client_id = dynamic[0].client_id
            if any(
                route.client_id == client_id
                for item in self._registrations.values()
                for route in item.execution_routes.values()
            ):
                raise ValueError("动态路由需要独占执行客户端")
        else:
            dynamic_clients = {
                route.client_id
                for item in self._registrations.values()
                for route in item.execution_routes.values()
                if isinstance(route, DynamicExecutionRoute)
            }
            if any(route.client_id in dynamic_clients for route in execution_routes):
                raise ValueError("执行客户端已被动态路由独占")
        self._registrations[strategy.strategy_id] = _Registration(
            strategy=strategy,
            data_bindings=tuple(data_bindings),
            execution_routes=routes,
            time_feed_ids=tuple(time_feed_ids),
        )
        self._has_dynamic_routes = self._has_dynamic_routes or bool(dynamic)
        self._market_health_gate.register_strategy(
            strategy.strategy_id,
            frozenset(binding.feed_id for binding in data_bindings),
        )

    def start(self) -> None:
        if self._started:
            return
        self._validate_configuration()
        self._bindings.clear()
        for client in self._clients.values():
            register_events = getattr(client, "register_execution_event_handler", None)
            if callable(register_events):
                register_events(self._on_execution_event)
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

        dynamic_route = next(
            (route for route in registration.execution_routes.values()
             if isinstance(route, DynamicExecutionRoute)),
            None,
        )
        if dynamic_route is not None:
            self._submit_dynamic_locked(intent, dynamic_route)
            return

        # 行情闸门必须先于TargetStore执行。被拒绝的目标不能占用revision或污染
        # 组合状态，否则行情恢复后同一目标无法安全重试。
        previous_target = self._target_store.get(intent.strategy_id)
        gate_decision = self._market_health_gate.check_target(intent, previous_target)

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
                        "market_health_mode": gate_decision.access_mode.value,
                        "market_health_state": gate_decision.snapshot.state.value,
                        "market_health_feeds": tuple(
                            sorted(
                                set(gate_decision.snapshot.unhealthy_feeds)
                                | set(gate_decision.snapshot.affected_feeds),
                            ),
                        ),
                    },
                ),
            )

    def _submit_dynamic_locked(self, intent: TargetPortfolio, route: DynamicExecutionRoute) -> None:
        # 先验证角色及as-of可用性，再提交TargetStore，避免无合约时污染目标版本。
        selection = route.resolver.resolve(route.target_key, intent.ts_event)
        if selection.target_key != route.target_key:
            raise ValueError("合约解析器返回的target_key与动态路由不匹配")
        previous = self._target_store.get(intent.strategy_id)
        self._market_health_gate.check_target(intent, previous)
        prior_targets = self._target_store.all()
        materialized = self._target_store.apply(intent)
        try:
            if self._publishing_strategy_event:
                state = self._roll_coordinator.state(intent.strategy_id, route.target_key)
                if state is None or state.phase is not RollPhase.ACTIVE or state.active.instrument_id != selection.instrument_id:
                    old_id = None if state is None else state.active.instrument_id
                    old_exposure = old_id is not None and (
                        self._position_manager.account_position(route.client_id, old_id) != 0
                        or self._position_manager.working_quantity(route.client_id, old_id) != 0
                    )
                    required_id = old_id if old_exposure else selection.instrument_id
                    if self._latest_market_event_ns.get(required_id) != intent.ts_event:
                        # 保留新逻辑目标，等待所需真实合约的新行情再推进换月。
                        return
            self._drive_dynamic_transaction_locked(materialized, route, selection, intent.ts_event)
        except Exception:
            self._target_store.restore(prior_targets)
            raise

    def refresh_dynamic_routes(
        self, as_of_ns: int, *, trigger_instrument_id: InstrumentId | None = None,
    ) -> None:
        """推进换月；行情驱动时只在待交易合约收到新行情后推进。"""
        if as_of_ns < 0:
            raise ValueError("as_of_ns不能为负数")
        with self._submit_lock:
            for strategy_id, registration in self._registrations.items():
                route = next(
                    (item for item in registration.execution_routes.values()
                     if isinstance(item, DynamicExecutionRoute)),
                    None,
                )
                materialized = self._target_store.get(strategy_id)
                if route is None or materialized is None:
                    continue
                selection = route.resolver.resolve(route.target_key, as_of_ns)
                if selection.target_key != route.target_key:
                    raise ValueError("合约解析器返回的target_key与动态路由不匹配")
                state = self._roll_coordinator.state(strategy_id, route.target_key)
                if state is not None and state.phase is RollPhase.ACTIVE and state.active.instrument_id == selection.instrument_id:
                    continue
                if trigger_instrument_id is not None:
                    old_id = None if state is None else state.active.instrument_id
                    old_exposure = old_id is not None and (
                        self._position_manager.account_position(route.client_id, old_id) != 0
                        or self._position_manager.working_quantity(route.client_id, old_id) != 0
                    )
                    # 有旧仓时只在旧合约行情到达后撤单/平仓；旧仓归零后只在
                    # 新合约行情到达后开仓。其他品种的Bar不得触发陈旧价订单。
                    required_id = old_id if old_exposure else selection.instrument_id
                    if trigger_instrument_id != required_id:
                        continue
                self._drive_dynamic_transaction_locked(materialized, route, selection, as_of_ns)

    def _drive_dynamic_transaction_locked(self, materialized: TargetPortfolio, route: DynamicExecutionRoute, selection, now_ns: int) -> None:
        prior_roll = self._roll_coordinator.snapshot()
        prior_portfolio = self._portfolio_coordinator.state()
        try:
            self._drive_dynamic_locked(materialized, route, selection, now_ns)
        except Exception:
            # 发单/撤单失败后的外部状态不一定可逆；本地恢复并关闭实盘闸门。
            self._roll_coordinator.restore(prior_roll)
            self._portfolio_coordinator.restore(prior_portfolio)
            self._position_manager.mark_recovery_required(route.client_id)
            raise

    def _drive_dynamic_locked(self, materialized: TargetPortfolio, route: DynamicExecutionRoute, selection, now_ns: int) -> None:
        current = self._roll_coordinator.state(materialized.strategy_id, route.target_key)
        if current is None:
            account = self._position_manager.snapshot()
            if any(
                key.client_id == route.client_id and quantity != 0
                for values in (account.account_positions, account.working_quantities)
                for key, quantity in values.items()
            ):
                raise RuntimeError("动态路由首次启动要求独占账户为空；已有仓位须先恢复换月状态")
        old_id = None if current is None else current.active.instrument_id
        position = Decimal(0) if old_id is None else self._position_manager.account_position(route.client_id, old_id)
        working = Decimal(0) if old_id is None else self._position_manager.working_quantity(route.client_id, old_id)
        healthy = self._market_health_gate.snapshot(materialized.strategy_id).access_mode.value == "NORMAL"
        target_unexpired = materialized.deadline_ns is None or now_ns <= materialized.deadline_ns
        reconciled = (
            not self._position_manager.is_recovery_required(route.client_id)
            and (self.mode is not RuntimeMode.LIVE
            or (self._position_manager.is_account_reconciled(route.client_id)
                and not self._position_manager.is_recovery_required(route.client_id)))
        )
        decision = self._roll_coordinator.step(
            materialized.strategy_id,
            selection,
            materialized.targets[route.target_key],
            old_position=position,
            old_working=working,
            allow_open=healthy and reconciled and target_unexpired,
        )
        client = self._clients[route.client_id]
        if decision.cancel_strategy_orders:
            client.cancel_strategy(materialized.strategy_id)
        if decision.targets is None:
            return
        current_revision = self._portfolio_coordinator.state().strategy_revisions.get(materialized.strategy_id, 0)
        revision = max(materialized.revision, current_revision + 1)
        resolved = {AccountTargetKey(route.client_id, item): quantity for item, quantity in decision.targets.items()}
        snapshot = self._portfolio_coordinator.update(
            strategy_id=materialized.strategy_id,
            revision=revision,
            ts_event=now_ns,
            targets=resolved,
        )
        client.submit_targets(ExecutionRequest(
            strategy_id=materialized.strategy_id,
            revision=revision,
            client_id=route.client_id,
            ts_event=now_ns,
            targets={
                key.instrument_id: quantity
                for key, quantity in snapshot.targets.items()
                if key.client_id == route.client_id
            },
            logical_targets=materialized.targets,
            execution_policy=materialized.execution_policy,
            deadline_ns=materialized.deadline_ns,
            metadata={
                **materialized.metadata,
                "portfolio_revision": snapshot.revision,
                "contract_revision": selection.revision,
                "roll_phase": decision.phase.value,
                "signal_ts_event": materialized.ts_event,
                "market_health_mode": self._market_health_gate.snapshot(materialized.strategy_id).access_mode.value,
            },
        ))

    def market_health_snapshot(self, strategy_id: str):
        return self._market_health_gate.snapshot(strategy_id)

    def confirm_market_recovery(
        self,
        strategy_id: str,
        *,
        operator: str,
        reason: str,
    ) -> RecoveryConfirmation:
        return self._market_health_gate.confirm_recovery(
            strategy_id,
            operator=operator,
            reason=reason,
        )

    def position(self, strategy_id: str, target_key: str) -> Decimal:
        registration = self._registrations.get(strategy_id)
        if registration is None or target_key not in registration.execution_routes:
            raise ValueError(f"未知策略目标: {strategy_id}/{target_key}")
        return self._position_provider.position(strategy_id, target_key)

    def _static_execution_route(self, strategy_id: str, target_key: str) -> ExecutionRoute:
        registration = self._registrations.get(strategy_id)
        route = None if registration is None else registration.execution_routes.get(target_key)
        if route is None:
            raise ValueError(f"未知策略目标: {strategy_id}/{target_key}")
        if not isinstance(route, ExecutionRoute):
            raise ValueError("动态路由的账户仓位可能涉及新旧合约，请按真实合约查询")
        return route

    def account_position(self, strategy_id: str, target_key: str) -> Decimal:
        route = self._static_execution_route(strategy_id, target_key)
        return self._position_manager.account_position(route.client_id, route.instrument_id)

    def working_quantity(self, strategy_id: str, target_key: str) -> Decimal:
        route = self._static_execution_route(strategy_id, target_key)
        return self._position_manager.working_quantity(route.client_id, route.instrument_id)

    def _on_execution_event(self, event: OrderUpdateEvent | FillEvent) -> None:
        """仅接受本Runner已注册客户端和策略的标准执行事件。"""
        identity = event.identity
        registration = self._registrations.get(identity.strategy_id)
        if identity.client_id not in self._clients or registration is None:
            raise ValueError("执行事件客户端或策略未注册")
        if not any(
            route.client_id == identity.client_id
            and (not isinstance(route, ExecutionRoute)
                 or str(route.instrument_id) == identity.instrument_id)
            for route in registration.execution_routes.values()
        ):
            raise ValueError("执行事件不属于该策略的交易路由")
        if not registration.strategy.is_started:
            raise RuntimeError("策略尚未启动，执行事件不能静默丢弃")
        registration.strategy._handle_execution_event(event)

    def publish(self, feed_id: str, event: Any) -> None:
        """发布标准行情事件，供行情适配器和契约测试使用。"""
        data_type = _event_data_type(event)
        instrument_id = _event_instrument_id(event)
        self._latest_market_event_ns[instrument_id] = event.ts_event
        for observer in tuple(self._market_observers):
            observer.on_market_event(event)
        if self._has_dynamic_routes:
            self.refresh_dynamic_routes(event.ts_event, trigger_instrument_id=instrument_id)
        key = (feed_id, data_type, instrument_id)
        self._publishing_strategy_event = True
        try:
            for strategy_id, binding in tuple(self._bindings.get(key, ())):
                if not _bar_spec_matches(binding, event):
                    continue
                self._registrations[strategy_id].strategy._handle_event(binding.data_key, event)
        finally:
            self._publishing_strategy_event = False

    def publish_time(self, feed_id: str, ts_event: int) -> None:
        """把独立时钟事件分发给显式绑定的策略。"""
        if not self._started:
            return
        if ts_event < 0:
            raise ValueError("时钟时间不能为负")
        for registration in tuple(self._registrations.values()):
            if feed_id in registration.time_feed_ids:
                registration.strategy.on_time(ts_event)

    def _validate_configuration(self) -> None:
        if not self._registrations:
            raise ValueError("没有配置策略")
        for strategy_id, registration in self._registrations.items():
            for binding in registration.data_bindings:
                if binding.feed_id not in self._feeds:
                    raise ValueError(
                        f"策略 {strategy_id} 引用了未知行情源 {binding.feed_id}",
                    )
            for feed_id in registration.time_feed_ids:
                feed = self._feeds.get(feed_id)
                if feed is None or not callable(getattr(feed, "register_time_handler", None)):
                    raise ValueError(f"策略 {strategy_id} 引用了无时钟能力的 feed {feed_id}")
            for route in registration.execution_routes.values():
                if route.client_id not in self._clients:
                    raise ValueError(
                        f"策略 {strategy_id} 引用了未知交易客户端 {route.client_id}",
                    )

    def _attach_feed(self, feed_id: str, feed: MarketDataFeed) -> None:
        register_health = getattr(feed, "register_health_handler", None)
        if callable(register_health) and feed_id not in self._attached_health_feeds:
            register_health(
                lambda snapshot, fid=feed_id: self._market_health_gate.on_feed_health(
                    fid,
                    snapshot,
                ),
            )
            self._attached_health_feeds.add(feed_id)
        if feed_id not in self._attached_feeds:
            register_time = getattr(feed, "register_time_handler", None)
            if callable(register_time):
                register_time(lambda ts, fid=feed_id: self.publish_time(fid, ts))
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
