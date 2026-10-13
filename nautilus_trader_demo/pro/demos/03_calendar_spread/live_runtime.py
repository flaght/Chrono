"""固定双腿Live装配；同一分钟两腿到齐才分派原策略。"""

from dataclasses import asdict, replace
from decimal import Decimal
import time
from types import SimpleNamespace

from bomber.framework.market.basic.base import Bar, DataType
from bomber.framework.trader.assembly import StrategyBindings, assemble_strategy
from bomber.framework.trader.contracts import DataBinding, ExecutionRoute, RuntimeMode
from bomber.framework.trader.portfolio import PositionManager
from bomber.framework.trader.runner import UnifiedStrategyRunner
from bomber.framework.trader.execution.builders import (ExecutionComponents, build_model_execution,
    build_ctp_portfolio_execution, build_fanout_execution, attach_execution_persistence, attach_ctp_persistence)
from bomber.framework.trader.execution.policy import execution_policy_from_args
from bomber.framework.trader.execution.recording import RecordingExecutionClient
from bomber.framework.trader.execution.risk import RiskLimits, MarketReferencePriceStore
from bomber.framework.trader.runtime.live.channels.ctp import (
    CtpLiveProfile, CtpMarketChannel, CtpExecutionChannel, InstrumentHealthGate)
from bomber.framework.trader.runtime.live.contracts import SessionReadiness
from bomber.framework.trader.runtime.live.local import LocalExecutionChannel
from bomber.framework.trader.runtime.live.profiles import build_minute_feed
from bomber.framework.trader.runtime.live.health import validate_instrument_max_age
from bomber.framework.trader.runtime.live.recovery import (
    InstrumentPositionScope, StrategyPositionBinding, SessionIdentityCheckpoint)
from .strategy import CalendarSelection, CalendarSpreadConfig, CalendarSpreadStrategy

CLIENT_ID = "calendar-spread-live"
MINUTE = 60_000_000_000
MAX_BAR_AGE = 120_000_000_000


def restore_direction(session):
    target = session.runner.target_store.get(session.strategy.strategy_id)
    if target is None:
        return 0
    ids = tuple(session.runner.instruments.values())
    first, second = (target.targets[str(i)] for i in ids)
    quantity = session.strategy.config.quantity
    if first != -second or first not in {-quantity, Decimal(0), quantity}:
        raise RuntimeError("恢复目标不是本策略的等手数反向双腿")
    return int(first / quantity)


class CalendarSessionRunner(UnifiedStrategyRunner):
    bound_instrument = None

    def __init__(self, references, positions, *, orders, clock_ns=None, instrument_max_age_seconds=10):
        super().__init__(RuntimeMode.LIVE, position_manager=positions)
        self.references, self.orders = references, orders
        self._clock_ns = clock_ns or time.time_ns
        self.instrument_max_age_ns = validate_instrument_max_age(instrument_max_age_seconds)
        self.expected_day = references.trading_day.strftime("%Y%m%d")
        self.fixed_spec = references.spec
        self.fixed_reference_fingerprint = references.manifest["fingerprint"]
        self.instruments = dict(references.instrument_ids)
        self.fixed_ids = frozenset(self.instruments.values())
        self.accept_bars = False
        self.failure = None
        self.first_complete_bar_ns = 0
        self.pending_frames = {}
        self.bar_counts = {i: 0 for i in self.fixed_ids}
        self.tick_events, self.tick_receives = {}, {}
        self.expired_frame_count = 0
        self.strategy = self.client = self.manager = None
        self.instrument_health = self.readiness = None

    def reference_is_current(self):
        return (self.references.spec == self.fixed_spec and
                self.references.manifest["fingerprint"] == self.fixed_reference_fingerprint)

    def stale_instruments(self):
        return {} if self.instrument_health is None else self.instrument_health.stale_instruments()

    def session_ready(self):
        return not self.failure and self.reference_is_current() and bool(self.readiness())

    def begin_bars(self):
        with self._submit_lock:
            if self.strategy._revision:
                session = SimpleNamespace(runner=self, strategy=self.strategy)
                self.strategy.rewarm_after_restore(restore_direction(session))
            # 第一个不完整分钟不进入信号窗口。
            now = self._clock_ns()
            self.first_complete_bar_ns = ((now + MINUTE - 1) // MINUTE + 1) * MINUTE - 1
            self.pending_frames.clear()
            self.accept_bars = True

    def close_role_gate(self, reason):
        self.failure, self.accept_bars = str(reason), False
        if self.orders:
            disarm = getattr(self.client, "disarm", None)
            if callable(disarm):
                disarm("calendar_spread_failed")
            self.position_manager.mark_recovery_required(self.client.client_id)
        raise RuntimeError(reason)

    def _check(self):
        if not self.session_ready():
            raise RuntimeError("双腿行情、参考资料或执行渠道失效")
        failure = getattr(self.client, "portfolio_failure", None)
        if failure or getattr(self.client, "report_errors", ()):
            raise RuntimeError(failure or "执行回报冲突")

    def _submit_locked(self, intent):
        self._check()
        previous = self.target_store.get(intent.strategy_id)
        if (self.orders and previous is not None and dict(previous.targets) != dict(intent.targets)
                and any(self.position_manager.working_quantity(CLIENT_ID, instrument)
                        for instrument in self.fixed_ids)):
            # 旧限价单不能继续追逐已被替换的目标。撤单请求不代表已撤单；
            # 公共Planner会等待真实撤单/成交回报释放在途，再由新Tick推进。
            self.client.cancel_strategy(intent.strategy_id)
            self._check()
        intent = replace(intent, ts_event=self._clock_ns(), metadata={**intent.metadata,
            "signal_bar_ns": intent.ts_event,
            "simnow_environment": self.market_profile.environment,
            "replay_md_trading_day": self.market_profile.replay_md_trading_day})
        return super()._submit_locked(intent)

    def observe_tick(self, tick):
        if tick.instrument_id not in self.fixed_ids:
            return
        with self._submit_lock:
            if tick.ts_event < self.tick_events.get(tick.instrument_id, -1):
                self.close_role_gate("双腿Tick时间回退")
            self.tick_events[tick.instrument_id] = tick.ts_event
            self.tick_receives[tick.instrument_id] = time.monotonic_ns()

    def continue_on_tick(self, tick):
        with self._submit_lock:
            if (not self.orders or not self.accept_bars or tick.instrument_id not in self.fixed_ids
                    or self.strategy.last_targets is None):
                return
            if not 0 <= self._clock_ns() - tick.ts_event <= self.instrument_max_age_ns:
                return
            try:
                self._check()
                super().publish("ctp-execution-ticks", tick)
                self.continue_execution_target(self.strategy.strategy_id, str(tick.instrument_id),
                    tick.ts_event, trigger_instrument_id=tick.instrument_id)
            except Exception as error:
                self.close_role_gate(str(error))

    def publish(self, feed_id, event):
        if not isinstance(event, Bar) or not self.accept_bars:
            return
        instrument = event.bar_type.instrument_id
        if instrument not in self.fixed_ids or event.ts_event < self.first_complete_bar_ns:
            return
        with self._submit_lock:
            try:
                if feed_id != "ctp-bars":
                    raise ValueError("跨期Bar来源与绑定不一致")
                self._check()
                now, stamp = self._clock_ns(), event.ts_event
                if stamp > now:
                    raise RuntimeError("双腿Bar来自未来")
                for pending in tuple(self.pending_frames):
                    if now - pending > MAX_BAR_AGE:
                        del self.pending_frames[pending]
                        self.expired_frame_count += 1
                if now - stamp > MAX_BAR_AGE or stamp <= self.strategy.last_frame_ns:
                    return
                frame = self.pending_frames.setdefault(stamp, {})
                if instrument in frame:
                    return
                frame[instrument] = event
                self.bar_counts[instrument] += 1
                for pending in sorted(tuple(self.pending_frames)):
                    frame = self.pending_frames[pending]
                    if pending <= self.strategy.last_frame_ns:
                        del self.pending_frames[pending]
                        continue
                    if set(frame) != self.fixed_ids:
                        continue
                    # 先将两腿价格同时更新给执行风控，再发布同步帧。
                    for bar in frame.values():
                        self.prices.on_market_event(bar)
                    for current in self.instruments.values():
                        self._check()
                        super().publish(feed_id, frame[current])
                    del self.pending_frames[pending]
                # 未完成目标由after_tick当前真实报价推进，Bar不推进开平今昨订单。
            except Exception as error:
                self.close_role_gate(str(error))


def assemble(args, references, driver, upstream, *, account_routes=(), clock_ns=None):
    references.validate()
    references.assert_expected(args.expected_instruments)
    model, orders = args.mode == "model", args.mode != "recording"
    if account_routes and not model:
        raise ValueError("跟随账户须选择model主策略账本")
    policy = execution_policy_from_args(args)
    profile = CtpLiveProfile(args.simnow_environment, args.replay_md_trading_day)
    positions = PositionManager()
    runner = CalendarSessionRunner(references, positions, orders=orders, clock_ns=clock_ns,
        instrument_max_age_seconds=args.instrument_max_age_seconds)
    runner.market_profile = profile
    runner.replay_md_trading_day = profile.replay_md_trading_day
    market = CtpMarketChannel(upstream, runner.expected_day, profile=profile)
    runner.instrument_health = InstrumentHealthGate(upstream, runner.fixed_ids, runner.expected_day,
        policy=profile, max_age_seconds=args.instrument_max_age_seconds, clock_ns=runner._clock_ns,
        fallback_events=runner.tick_events, fallback_receives=runner.tick_receives)
    market.instrument_health = runner.instrument_health
    ids, specs = references.instrument_ids, references.specs
    scope = InstrumentPositionScope(ids.values())
    limits = {ids[r]: RiskLimits(max_order_quantity=args.quantity * 2,
        max_abs_position=args.quantity, max_order_notional=args.max_notional,
        max_abs_position_notional=args.max_notional, max_market_age_ns=MAX_BAR_AGE,
        contract_multiplier=specs[r].multiplier) for r in references.roles}
    if model:
        execution = build_model_execution(CLIENT_ID, instrument_limits=limits, positions=positions)
    elif orders:
        execution = build_ctp_portfolio_execution(driver, trading_day=runner.expected_day,
            price_increments={ids[r]: specs[r].tick for r in references.roles},
            multipliers={ids[r]: specs[r].multiplier for r in references.roles},
            instrument_limits=limits, session_check=runner.session_ready, orders=True,
            positions=positions, clock_ns=runner._clock_ns,
            limit_offset_ticks=args.limit_offset_ticks, instrument_scope=scope,
            account_reconciliation=policy.account_reconciliation,
            generic_close_venues=("DCE", "CZCE"), generic_close_today_first_venues=("DCE", "CZCE"))
    else:
        execution = ExecutionComponents(RecordingExecutionClient(CLIENT_ID), positions, MarketReferencePriceStore())
    if account_routes:
        execution = build_fanout_execution(execution, account_routes)
    execution_channel = (CtpExecutionChannel(driver, runner.expected_day, require_orders=True) if args.mode == "simnow"
        else LocalExecutionChannel(CLIENT_ID, lambda: execution.client.is_started if not model
            else execution.client.is_reconciled, channel=args.mode))
    runner.readiness = SessionReadiness(market, execution_channel)
    config = CalendarSpreadConfig(args.leg1_role, args.leg2_role, args.quantity, args.lookback,
        args.entry_z, args.exit_z, args.rebalance_interval)
    symbols = tuple(str(ids[r].symbol).lower() for r in references.roles)
    strategy = CalendarSpreadStrategy(CLIENT_ID, config,
        (CalendarSelection(references.trading_day, 0, *symbols),), dict(zip(symbols, ids.values())))
    runner.strategy, runner.client, runner.prices = strategy, execution.client, execution.prices
    references.publication_lock = runner._submit_lock
    upstream.register_trade_tick_handler(runner.observe_tick)
    feed = build_minute_feed("CALENDAR_SPREAD_1M", upstream, profile,
        after_tick=runner.continue_on_tick, max_age_seconds=args.instrument_max_age_seconds, clock_ns=runner._clock_ns)
    for role in references.roles:
        feed.register_instrument(references.instrument_meta(role))
    bindings = StrategyBindings(
        data=tuple(DataBinding(str(i), "ctp-bars", i, DataType.BAR, "1-MINUTE") for i in ids.values()),
        execution=tuple(ExecutionRoute(str(i), CLIENT_ID, i) for i in ids.values()))
    assemble_strategy(runner, strategy, feeds={"ctp-bars": feed}, execution=execution, bindings=bindings)
    identity = {"mode": args.mode, "trading_day": runner.expected_day,
        "simnow_environment": profile.environment, "replay_md_trading_day": profile.replay_md_trading_day,
        "product": references.product, "roles": list(references.roles),
        "instruments": {r: str(i) for r, i in ids.items()},
        "reference_fingerprint": references.manifest["fingerprint"],
        "config": {k: str(v) for k, v in asdict(config).items()},
        "max_notional": str(args.max_notional), "limit_offset_ticks": args.limit_offset_ticks,
        "instrument_max_age_seconds": args.instrument_max_age_seconds,
        "execution_policy": policy.snapshot_state()}
    attach = attach_execution_persistence if model else attach_ctp_persistence
    manager = attach(execution, runner, state_file=args.state_file, state_components={
        "calendar_session": SessionIdentityCheckpoint(identity)}) if orders else None
    runner.manager = manager
    return SimpleNamespace(runner=runner, strategy=strategy, client=execution.client, execution=execution,
        ledger=execution.ledger, manager=manager, driver=driver, references=references, upstream=upstream,
        bar_feed=feed, market_profile=profile, readiness=runner.readiness, position_scope=scope,
        recovery_binding=StrategyPositionBinding(CLIENT_ID, CLIENT_ID, {str(i): i for i in ids.values()}))
