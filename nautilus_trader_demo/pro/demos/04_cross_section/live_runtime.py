"""04固定多主力会话装配；算法仍由原strategy.py提供。"""

from dataclasses import replace
from types import SimpleNamespace
import time

from bomber.framework.market.basic.base import Bar, InstrumentId, DataType
from bomber.framework.trader import DataBinding, ExecutionRoute, PositionManager, RiskLimits, RuntimeMode
from bomber.framework.trader.runner import UnifiedStrategyRunner
from bomber.framework.trader.assembly import StrategyBindings, assemble_strategy
from bomber.framework.trader.execution.builders import build_ctp_portfolio_execution, attach_ctp_persistence
from bomber.framework.trader.runtime.live.channels.ctp import CtpSessionLifecycle
from bomber.framework.dataprep.live_portfolio import PortfolioRoleReferences as PortfolioReferences
from bomber.framework.trader.runtime.live.channels.ctp import (InstrumentHealthGate, CtpLiveProfile,
    CtpMarketChannel, CtpExecutionChannel)
from bomber.framework.trader.runtime.live.contracts import SessionReadiness
from bomber.framework.trader.runtime.live.recovery import StrategyPositionBinding, InstrumentPositionScope
from bomber.framework.trader.runtime.live.health import validate_instrument_max_age
from bomber.framework.trader.runtime.live.profiles import build_minute_feed
from .recovery import ResumeLifecycle

from .strategy import MainCrossSectionMomentumStrategy

CLIENT_ID = "cross-section-simnow"
MINUTE = 60_000_000_000




class CrossSectionSessionRunner(UnifiedStrategyRunner):
    """完整同步帧决策后，以本帧各腿真实行情继续未完成的固定目标。"""
    bound_instrument = None  # 单合约生命周期旧报告字段；组合报告另列全部合约。

    def _submit_locked(self, intent):
        intent = replace(intent, metadata={**intent.metadata,
            **self.session_metadata})
        # 信号属于已收盘分钟；风控估值应与实际执行时刻比较。
        if self.orders:
            intent = replace(intent, ts_event=self._clock_ns(),
                metadata={**intent.metadata, "signal_bar_ns": intent.ts_event})
        return super()._submit_locked(intent)

    def __init__(self, *, references, session_check, orders, position_manager, clock_ns=None,
                 instrument_max_age_seconds=10, session_metadata=None):
        super().__init__(RuntimeMode.LIVE, position_manager=position_manager)
        self.references, self.orders = references, orders
        self._session_check = session_check
        self._clock_ns = clock_ns or (lambda: time.time_ns())
        self.instrument_max_age_ns = validate_instrument_max_age(instrument_max_age_seconds)
        self.expected_day = references.trading_day.strftime("%Y%m%d")
        self.session_metadata = dict(session_metadata or {})
        self.instrument_health = None
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
            stamp = tick.ts_event
            if stamp <= 0 or not 0 <= self._clock_ns() - stamp <= self.instrument_max_age_ns:
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
        if self.instrument_health is None:
            return {}
        failures = self.instrument_health.stale_instruments()
        self.depth_records = dict(self.instrument_health.depth_records)
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
                    if self.orders and self.strategy.last_targets is not None:
                        for current in self.instruments.values():
                            if not self.session_ready() or getattr(self.client, "portfolio_failure", None):
                                raise RuntimeError("逐腿推进期间会话失败或出现拒单")
                            self.continue_execution_target(self.strategy.strategy_id, str(current),
                                stamp, trigger_instrument_id=current)
            except Exception as error:
                self.close_role_gate(str(error))


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
        return super().poll(session)

    def shutdown(self, session, context):
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
    environment = getattr(args, "simnow_environment", "realtime")
    replay_day = getattr(args, "replay_md_trading_day", None)
    policy = CtpLiveProfile(environment, replay_day)
    expected_day = references.trading_day.strftime("%Y%m%d")
    positions = PositionManager()
    market_channel = CtpMarketChannel(upstream, expected_day, profile=policy)
    check = SessionReadiness(market_channel,
        CtpExecutionChannel(driver, expected_day, require_orders=orders))
    runner = CrossSectionSessionRunner(references=references, orders=orders,
        instrument_max_age_seconds=getattr(args, "instrument_max_age_seconds", 10),
        session_metadata={"simnow_environment": policy.environment, "replay_md_trading_day": replay_day},
        position_manager=positions, session_check=check)
    runner.replay_md_trading_day = replay_day  # 旧检查点/报告兼容字段，非策略输入。
    runner.expected_md_day = policy.expected_md_day(expected_day)
    runner.upstream = upstream
    runner.instrument_health = InstrumentHealthGate(upstream, runner.fixed_ids, expected_day,
        policy=policy, max_age_seconds=getattr(args, "instrument_max_age_seconds", 10),
        clock_ns=runner._clock_ns, fallback_events=runner.tick_events,
        fallback_receives=runner.tick_receives)
    market_channel.instrument_health = runner.instrument_health
    ids, specs = references.instrument_ids, references.specs
    position_scope = InstrumentPositionScope(ids.values())
    limits = {ids[p]: RiskLimits(max_order_quantity=args.max_quantity,
        max_abs_position=args.max_quantity, max_order_notional=args.max_notional,
        max_abs_position_notional=args.max_notional, max_market_age_ns=120_000_000_000,
        contract_multiplier=specs[p].multiplier) for p in args.products}
    execution = build_ctp_portfolio_execution(driver, trading_day=expected_day,
        price_increments={ids[p]: specs[p].tick for p in args.products},
        multipliers={ids[p]: specs[p].multiplier for p in args.products},
        instrument_limits=limits, session_check=runner.session_ready, orders=orders,
        positions=positions, limit_offset_ticks=args.limit_offset_ticks, instrument_scope=position_scope)
    strategy = MainCrossSectionMomentumStrategy(CLIENT_ID, products=args.products, roles=references,
        instruments={str(ids[p].symbol).lower(): ids[p] for p in args.products},
        multipliers={ids[p]: specs[p].multiplier for p in args.products},
        lookback=args.lookback, rebalance_interval=args.rebalance_interval,
        target_notional=args.target_notional, group_fraction=args.group_fraction,
        target_quantity_cap=getattr(args, "target_quantity_cap", None),
        require_full_groups=getattr(args, "require_full_groups", False))
    runner.strategy, runner.client = strategy, execution.client
    references.publication_lock = runner._submit_lock
    # 先记录各腿Tick新鲜度，再让聚合器分派由新Tick收盘的Bar。
    upstream.register_trade_tick_handler(runner.observe_tick)
    feed = build_minute_feed("CROSS_SECTION_1M", upstream,
        policy, after_tick=runner.continue_on_tick,
        max_age_seconds=getattr(args, "instrument_max_age_seconds", 10), clock_ns=runner._clock_ns)
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
        references=references, upstream=upstream, bar_feed=feed, market_profile=policy, readiness=check,
        position_scope=position_scope,
        recovery_binding=StrategyPositionBinding(strategy.strategy_id, execution.client.client_id,
            {str(i): i for i in ids.values()}))
