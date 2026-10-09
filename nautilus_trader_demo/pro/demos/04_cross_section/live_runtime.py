"""04固定多主力会话装配；算法仍由原strategy.py提供。"""

from datetime import datetime
from dataclasses import replace
from copy import deepcopy
from threading import RLock
from types import SimpleNamespace
import time

from bomber.framework.market.basic.base import Bar, InstrumentId, InstrumentMeta, DataType
from bomber.framework.market.stream import TradeTickBarFeed
from bomber.framework.trader import DataBinding, ExecutionRoute, PositionManager, RiskLimits, RuntimeMode
from bomber.framework.trader.runner import UnifiedStrategyRunner
from bomber.framework.trader.assembly import StrategyBindings, assemble_strategy
from bomber.framework.trader.execution.builders import build_ctp_portfolio_execution, attach_ctp_persistence
from bomber.framework.trader.runtime.ctp import CtpSessionLifecycle, session_ready
from .recovery import ResumeLifecycle

from .strategy import MainCrossSectionMomentumStrategy

CLIENT_ID = "cross-section-simnow"
MINUTE = 60_000_000_000


class PortfolioReferences:
    """适配已有多品种LiveFuturesReferences，固定本次路由及累计因子版本。"""
    paths = ()

    def __init__(self, service):
        self.service = service
        self.products = service.products
        self.trading_day = service.trading_day
        self.started_ns = service.started_ns
        self.factor_date_basis = service.factor_date_basis
        self._cache_lock = RLock()
        self.publication_lock = RLock()
        self._cache = None
        self._failure = None
        self._clock_high_water_ns = None
        self.refresh()

    def validate(self):
        specs = self.specs
        if any(s.venue not in {"SHFE", "INE"} or s.currency != "CNY" for s in specs.values()):
            raise ValueError("04首期SimNow仅支持SHFE/INE人民币期货")
        if len(set(self.instrument_ids.values())) != len(self.products):
            raise ValueError("多品种不能映射到同一个真实合约")

    @property
    def specs(self):
        with self._cache_lock:
            return dict(self._cache[1])

    @property
    def instrument_ids(self):
        return {p: InstrumentId.from_str(f"{s.symbol.lower()}.{s.venue}")
                for p, s in self.specs.items()}

    @property
    def spec(self):
        # 生命周期函数的固定资料比较包含因子，避免窗口混入新版本复权价。
        assignment = self.snapshot(time.time_ns())
        return tuple((p, assignment.instrument(p, "main"), self.specs[p],
                      assignment.factor(p, "main")) for p in self.products)

    @property
    def manifest(self):
        with self._cache_lock:
            result = deepcopy(self._cache[2]) if self._cache else {}
            failure = self._failure
            if failure is None and self._cache:
                now = time.time_ns()
                age = now - self._cache[2]["observed_at_ns"]
                limit = self._cache[2]["freshness_policy"]["max_observation_age_seconds"]
                if self._clock_high_water_ns is not None and now < self._clock_high_water_ns:
                    failure = self._failure = "参考缓存时钟回退"
                elif not 0 <= age <= limit * 1_000_000_000:
                    failure = "参考缓存观测过期或时钟回退"
                self._clock_high_water_ns = max(now, self._clock_high_water_ns or now)
            return {**result, "ready": failure is None and self._cache is not None,
                    "failure": failure}

    @property
    def factor_date(self):
        return datetime.fromisoformat(self.manifest["factor_date"]).date()

    def refresh(self):
        # 只由启动/控制线程调用。snapshot自身会刷新，不能先refresh再snapshot重复读库。
        try:
            assignment = self.service.snapshot(time.time_ns())
            specs = {p: self.service.instrument_specs[p]["main"] for p in self.products}
            manifest = deepcopy(self.service.manifest)
            if not manifest.get("ready", True):
                raise RuntimeError(manifest.get("failure") or "参考资料未就绪")
            manifest.setdefault("observed_at_ns", time.time_ns())
            manifest.setdefault("freshness_policy", {"max_observation_age_seconds": 30})
            # 发布缓存与策略发单互斥，但整个数据库查询过程不占用策略锁。
            with self.publication_lock:
                with self._cache_lock:
                    self._cache = (assignment, specs, manifest)
                    self._failure = None
                    self.validate()
        except Exception as error:
            with self._cache_lock:
                self._failure = str(error)
            raise

    def snapshot(self, as_of_ns):
        with self._cache_lock:
            manifest = self.manifest
            if not manifest["ready"]:
                raise RuntimeError(manifest["failure"])
            assignment = self._cache[0]
            if type(as_of_ns) is not int or as_of_ns < max(assignment.effective_ns, assignment.available_ns):
                raise RuntimeError("当前参考版本在Bar时刻尚不可见")
            return assignment

    def instrument_meta(self, product):
        s = self.specs[product]
        return InstrumentMeta(instrument_id=self.instrument_ids[product],
            price_precision=max(0, -s.tick.normalize().as_tuple().exponent), size_precision=0,
            price_increment=s.tick, multiplier=s.multiplier, currency=s.currency, exchange=s.venue)


class CrossSectionSessionRunner(UnifiedStrategyRunner):
    """完整同步帧决策后，以本帧各腿真实行情继续未完成的固定目标。"""
    bound_instrument = None  # 单合约生命周期旧报告字段；组合报告另列全部合约。

    def _submit_locked(self, intent):
        # 信号属于已收盘分钟；风控估值应与实际执行时刻比较。
        if self.orders:
            intent = replace(intent, ts_event=self._clock_ns(),
                metadata={**intent.metadata, "signal_bar_ns": intent.ts_event})
        return super()._submit_locked(intent)

    def __init__(self, *, references, session_check, orders, position_manager, clock_ns=None,
                 instrument_max_age_seconds=10):
        super().__init__(RuntimeMode.LIVE, position_manager=position_manager)
        self.references, self.orders = references, orders
        self._session_check = session_check
        self._clock_ns = clock_ns or (lambda: time.time_ns())
        if not 0 < instrument_max_age_seconds <= 120:
            raise ValueError("逐合约深度时效须在(0,120]秒内")
        self.instrument_max_age_ns = int(instrument_max_age_seconds * 1_000_000_000)
        self.expected_day = references.trading_day.strftime("%Y%m%d")
        self.fixed_spec = references.spec
        self.instruments = dict(references.instrument_ids)
        self.fixed_ids = frozenset(self.instruments.values())
        self.accept_bars = False
        self.failure = None
        self.first_complete_bar_ns = 0
        self.tick_events = {}
        self.tick_receives = {}
        self.last_bars = {}
        self.strategy = self.client = None
        self.upstream = None
        self.depth_records = {}
        self.expired_bar_count = 0
        self.expired_bar_samples = []
        self.pending_frames = {}
        self.bar_counts = {}
        self.latest_bar_ns = {}
        self.expired_frame_count = 0

    def bar_is_current(self, event):
        now = self._clock_ns()
        age = now - event.ts_event
        details = {"instrument": str(event.bar_type.instrument_id), "bar_ns": event.ts_event,
                   "now_ns": now, "age_seconds": age / 1e9}
        if age < 0:
            raise RuntimeError(f"分钟Bar来自未来: {details}")
        if age > 120_000_000_000:
            self.expired_bar_count += 1
            self.expired_bar_samples.append(details)
            self.expired_bar_samples = self.expired_bar_samples[-20:]
            print(f"跳过过期成交Bar: {details}", flush=True)
            return False
        return True

    def observe_tick(self, tick):
        if tick.instrument_id in self.fixed_ids:
            with self._submit_lock:
                if tick.ts_event < self.tick_events.get(tick.instrument_id, -1):
                    self.close_role_gate("单合约Tick时间回退")
                self.tick_events[tick.instrument_id] = tick.ts_event
                self.tick_receives[tick.instrument_id] = time.monotonic_ns()

    def continue_on_tick(self, tick):
        """聚合/策略决策之后，用当前合约真实Tick推进保留目标，不更新信号窗口。"""
        with self._submit_lock:
            if (not self.orders or not self.accept_bars or tick.instrument_id not in self.fixed_ids or
                    self.strategy.last_targets is None or
                    self.target_store.get(self.strategy.strategy_id) is None):
                return
            if not 0 <= self._clock_ns() - tick.ts_event <= self.instrument_max_age_ns:
                return
            try:
                if not self.session_ready() or self.references.spec != self.fixed_spec:
                    raise RuntimeError("Tick目标推进时行情或参考版本失效")
                if getattr(self.client, "portfolio_failure", None):
                    raise RuntimeError(self.client.portfolio_failure)
                super().publish("ctp-execution-ticks", tick)
                self.continue_execution_target(self.strategy.strategy_id, str(tick.instrument_id),
                    tick.ts_event, trigger_instrument_id=tick.instrument_id)
            except Exception as error:
                self.close_role_gate(str(error))

    def session_ready(self):
        if self.failure or not self._session_check():
            return False
        return not self.stale_instruments()

    def stale_instruments(self):
        now, received = self._clock_ns(), time.monotonic_ns()
        observations = getattr(self.upstream, "depth_observations", None)
        if observations:
            self.depth_records = dict(observations)
        failures = {}
        for i in self.fixed_ids:
            # 手动事件夹具没有原始CTP深度接口；生产CTP必须使用快照接收记录。
            observation = observations.get(i) if observations is not None else None
            event = (observation.ts_event if observation else
                     self.tick_events.get(i) if observations is None else None)
            receive = (observation.received_monotonic_ns if observation else
                       self.tick_receives.get(i) if observations is None else None)
            day_ok = observations is None or (observation and observation.trading_day == self.expected_day)
            regressed = bool(observation and observation.timestamp_regressed)
            event_age = None if event is None else now - event
            receive_age = None if receive is None else received - receive
            if (not day_ok or regressed or event_age is None or receive_age is None or
                    not 0 <= event_age <= self.instrument_max_age_ns or
                    not 0 <= receive_age <= self.instrument_max_age_ns):
                failures[str(i)] = {"event_age_seconds": None if event_age is None else event_age / 1e9,
                    "receive_age_seconds": None if receive_age is None else receive_age / 1e9,
                    "trading_day": observation.trading_day if observation else None,
                    "max_age_seconds": self.instrument_max_age_ns / 1e9,
                    "timestamp_regressed": regressed}
        return failures

    def begin_bars(self):
        self.first_complete_bar_ns = ((self._clock_ns() + MINUTE - 1) // MINUTE + 1) * MINUTE - 1
        self.accept_bars = True

    def close_role_gate(self, reason):
        self.failure = str(reason)
        self.accept_bars = False
        if self.orders and self.client is not None:
            self.position_manager.mark_recovery_required(self.client.client_id)
            self.client.disarm("cross_section_failed")
        raise RuntimeError(reason)

    def publish(self, feed_id, event):
        if not isinstance(event, Bar) or not self.accept_bars:
            return
        instrument = event.bar_type.instrument_id
        if instrument not in self.fixed_ids or event.ts_event < self.first_complete_bar_ns:
            return
        with self._submit_lock:
            now = self._clock_ns()
            for stamp in tuple(self.pending_frames):
                if now - stamp > 120_000_000_000:
                    del self.pending_frames[stamp]
                    self.expired_frame_count += 1
            if event.ts_event <= self.strategy.last_emitted_ns:
                return
            try:
                if not self.session_ready():
                    raise RuntimeError(f"任一品种行情陈旧或MD/TD会话失效: {self.stale_instruments()}")
                if not self.bar_is_current(event):
                    return
                if self.references.spec != self.fixed_spec:
                    raise RuntimeError("主力/条款/累计因子变化，固定会话须重新核对")
                if not self.session_ready():
                    raise RuntimeError("参考刷新后行情失效")
                if not self.bar_is_current(event):
                    return
                frame = self.pending_frames.setdefault(event.ts_event, {})
                if instrument in frame:
                    return
                frame[instrument] = (feed_id, event)
                self.bar_counts[instrument] = self.bar_counts.get(instrument, 0) + 1
                self.latest_bar_ns[instrument] = max(event.ts_event, self.latest_bar_ns.get(instrument, -1))
                for stamp in sorted(tuple(self.pending_frames)):
                    current_frame = self.pending_frames[stamp]
                    if stamp <= self.strategy.last_emitted_ns:
                        del self.pending_frames[stamp]
                        continue
                    if set(current_frame) != self.fixed_ids:
                        continue
                    if not self.bar_is_current(next(iter(current_frame.values()))[1]):
                        del self.pending_frames[stamp]
                        self.expired_frame_count += 1
                        continue
                    # 只发布同一分钟完整五腿；不会让下一分钟抢先覆盖较慢腿的当前分钟。
                    for current in self.instruments.values():
                        fid, bar = current_frame[current]
                        if not self.session_ready() or self.references.spec != self.fixed_spec:
                            raise RuntimeError("完整帧发布期间行情或参考版本失效")
                        super().publish(fid, bar)
                        self.last_bars[current] = stamp
                    del self.pending_frames[stamp]
                    if self.orders:
                        for current in self.instruments.values():
                            if not self.session_ready() or getattr(self.client, "portfolio_failure", None):
                                raise RuntimeError("逐腿推进期间会话失败或出现拒单")
                            self.continue_execution_target(self.strategy.strategy_id, str(current),
                                stamp, trigger_instrument_id=current)
            except Exception as error:
                self.close_role_gate(str(error))


class _CrossSectionBarFeed(TradeTickBarFeed):
    def __init__(self, *args, runner, **kwargs):
        super().__init__(*args, **kwargs)
        self.runner = runner

    def _on_trade_tick(self, tick):
        super()._on_trade_tick(tick)
        self.runner.continue_on_tick(tick)


class CrossSectionLifecycle(ResumeLifecycle, CtpSessionLifecycle):
    @staticmethod
    def _expected_gross(session):
        result = {}
        for instrument in session.runner.fixed_ids:
            state = session.ledger.snapshot(instrument)
            if state.long_total or state.short_total:
                result[str(instrument)] = (state.long_total, state.short_total)
        return result

    def poll(self, session):
        if self.orders and getattr(session.client, "portfolio_failure", None):
            session.runner.close_role_gate(session.client.portfolio_failure)
        stale = session.runner.stale_instruments()
        if stale:
            session.runner.close_role_gate(f"逐合约深度行情陈旧或失效: {stale}")
        session.references.refresh()
        return super().poll(session)

    def shutdown(self, session, context):
        if self.resuming and not self.resume_ready:
            # 校验失败不能保存未经核验的状态或撤销未知账户订单。
            if session:
                session.runner.accept_bars = False
                session.runner.stop()
            self.driver.stop()
            return {"final_gross": None, "final_active_orders": None,
                    "cleanup_errors": [], "resume_state_preserved": True}
        if session:
            with session.runner._submit_lock:
                session.runner.accept_bars = False
        return super().shutdown(session, context)

    def verify(self, session):
        required = session.strategy.lookback + 1
        if session.strategy.synchronized_frames < required:
            raise RuntimeError(f"同步帧预热不足: actual={session.strategy.synchronized_frames} required={required} "
                               f"bar_counts={session.runner.bar_counts}")
        if self.resuming:
            if session.client.report_errors or session.strategy.last_targets is None:
                raise RuntimeError("恢复后尚无有效新目标或存在回报错误")
        else:
            super().verify(session)
        if not self.orders:
            return
        positions = session.runner.position_manager
        for instrument in session.runner.fixed_ids:
            if positions.unassigned_position(CLIENT_ID, instrument) != 0:
                raise RuntimeError(f"成交存在未归属仓位: {instrument}")
        if not self.resuming and not session.strategy.fills_received:
            raise RuntimeError("没有策略成交回调，不能认定组合归属通过")
        for key, target in (session.strategy.last_targets or {}).items():
            instrument = InstrumentId.from_str(key)
            if (session.strategy.position(key) != target or
                    positions.working_quantity(CLIENT_ID, instrument) != 0):
                raise RuntimeError(f"组合最新目标尚未完成: {key} target={target} "
                                   f"position={session.strategy.position(key)}")

    def snapshot(self, session, context):
        fields = super().snapshot(session, context)
        fields.pop("fixed_instrument", None)
        return {**fields,
            "resume_ready": self.resume_ready,
            "restored_generation": getattr(self, "restored_generation", None),
            "fixed_instruments": ({p: str(i) for p, i in session.runner.instruments.items()} if session else {}),
            "per_instrument_tick_ns": ({str(i): t for i, t in session.runner.tick_events.items()} if session else {}),
            "per_instrument_depth": ({str(i): vars(o) for i, o in
                session.runner.depth_records.items()} if session else {}),
            "expired_bar_count": session.runner.expired_bar_count if session else 0,
            "expired_bar_samples": list(session.runner.expired_bar_samples) if session else [],
            "per_instrument_bar_count": {str(i): n for i, n in session.runner.bar_counts.items()} if session else {},
            "per_instrument_latest_bar_ns": {str(i): n for i, n in session.runner.latest_bar_ns.items()} if session else {},
            "expired_frame_count": session.runner.expired_frame_count if session else 0,
            "pending_frames": [{"bar_ns": stamp, "received": sorted(str(i) for i in frame),
                "missing": sorted(str(i) for i in session.runner.fixed_ids - set(frame))}
                for stamp, frame in sorted(session.runner.pending_frames.items())] if session else [],
            "portfolio_failure": getattr(session.client, "portfolio_failure", None) if session else None}


def assemble(args, references, driver, upstream):
    references.validate()
    actual = {p: str(i) for p, i in references.instrument_ids.items()}
    if args.expected_instruments and actual != args.expected_instruments:
        raise ValueError(f"本次主力不符: expected={args.expected_instruments} actual={actual}")
    orders = args.mode == "simnow"
    expected_day = references.trading_day.strftime("%Y%m%d")
    positions = PositionManager()
    runner = CrossSectionSessionRunner(references=references, orders=orders,
        instrument_max_age_seconds=getattr(args, "instrument_max_age_seconds", 10),
        position_manager=positions, session_check=lambda: session_ready(driver, upstream,
            expected_day, require_orders=orders))
    ids, specs = references.instrument_ids, references.specs
    limits = {ids[p]: RiskLimits(max_order_quantity=args.max_quantity,
        max_abs_position=args.max_quantity, max_order_notional=args.max_notional,
        max_abs_position_notional=args.max_notional, max_market_age_ns=120_000_000_000,
        contract_multiplier=specs[p].multiplier) for p in args.products}
    execution = build_ctp_portfolio_execution(driver, trading_day=expected_day,
        price_increments={ids[p]: specs[p].tick for p in args.products},
        multipliers={ids[p]: specs[p].multiplier for p in args.products},
        instrument_limits=limits, session_check=runner.session_ready, orders=orders,
        positions=positions, limit_offset_ticks=args.limit_offset_ticks)
    strategy = MainCrossSectionMomentumStrategy(CLIENT_ID, products=args.products, roles=references,
        instruments={str(ids[p].symbol).lower(): ids[p] for p in args.products},
        multipliers={ids[p]: specs[p].multiplier for p in args.products},
        lookback=args.lookback, rebalance_interval=args.rebalance_interval,
        target_notional=args.target_notional, group_fraction=args.group_fraction,
        target_quantity_cap=getattr(args, "target_quantity_cap", None),
        require_full_groups=getattr(args, "require_full_groups", False))
    runner.strategy, runner.client = strategy, execution.client
    references.publication_lock = runner._submit_lock
    runner.upstream = upstream
    # 先记录各腿Tick新鲜度，再让聚合器分派由新Tick收盘的Bar。
    upstream.register_trade_tick_handler(runner.observe_tick)
    feed = _CrossSectionBarFeed("CROSS_SECTION_1M", upstream, runner=runner)
    for p in args.products:
        feed.register_instrument(references.instrument_meta(p))
    bindings = StrategyBindings(
        data=tuple(DataBinding(str(i), "ctp-bars", i, DataType.BAR, "1-MINUTE") for i in ids.values()),
        execution=tuple(ExecutionRoute(str(i), CLIENT_ID, i) for i in ids.values()))
    assemble_strategy(runner, strategy, feeds={"ctp-bars": feed}, execution=execution, bindings=bindings)
    manager = attach_ctp_persistence(execution, runner, state_file=args.state_file) if orders else None
    runner.manager = manager
    return SimpleNamespace(runner=runner, strategy=strategy, client=execution.client,
        execution=execution, ledger=execution.ledger, manager=manager, driver=driver,
        references=references, upstream=upstream, bar_feed=feed)
