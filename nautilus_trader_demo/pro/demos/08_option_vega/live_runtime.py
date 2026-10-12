"""Vega会话的行情帧、约束和报告；网络连接由公共渠道模块负责。"""
from datetime import datetime
from dataclasses import replace
from decimal import Decimal
import threading
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from bomber.framework.market.basic.base import Bar, InstrumentId
from bomber.framework.market.frames import MinuteFrameBuffer, minute_close_ns
from bomber.framework.trader import RuntimeMode, UnifiedStrategyRunner
from bomber.framework.trader.execution.builders import (ExecutionComponents, build_model_execution,
    build_ctp_portfolio_execution, attach_execution_persistence, attach_ctp_persistence, build_fanout_execution)
from bomber.framework.trader.execution.recording import RecordingExecutionClient
from bomber.framework.trader.execution.policy import execution_policy_from_args
from bomber.framework.trader.portfolio import PositionManager
from bomber.framework.trader.execution.risk import MarketReferencePriceStore
from bomber.framework.trader.runtime.live.contracts import SessionReadiness
from bomber.framework.trader.runtime.live.recovery import (StrategyPositionBinding, InstrumentPositionScope,
    SessionIdentityCheckpoint)
from bomber.framework.trader.runtime.live.local import LocalExecutionChannel
from bomber.framework.trader.runtime.live.channels.ctp import (CtpLiveProfile, CtpMarketChannel,
    CtpExecutionChannel, InstrumentHealthGate)
from bomber.framework.trader.runtime.live.channels.dolphin import DolphinMarketChannel
from .assembly import assemble_strategy_components, risk_limits
from .checkpoint import OptionVegaCheckpoint
from .vega_signal import Contract

CLIENT_ID = "option-vega-live"


class OptionReferences:
    """公共条款快照投影成本策略关系，不将指数注册为可交易标的。"""
    refresh_on_event = False

    def __init__(self, service, trading_day, calendar, *, calendar_manifest=None):
        self.service, self.trading_day, self.calendar = service, trading_day, calendar
        self.calendar_manifest = dict(calendar_manifest or {})
        snap = service.snapshot()
        self.fingerprint = snap.fingerprint
        self.contracts = {key: Contract(key, service.future_product + spec.month[2:], float(spec.strike),
            spec.multiplier, spec.listed, spec.last_trade, spec.expiry, spec.tick)
            for key, spec in snap.options.items()}
        self.futures = {key: dict(row) for key, row in snap.futures.items()}
        self.instrument_ids = {key: InstrumentId.from_str(key + ".CFFEX")
            for key in (*self.contracts, *self.futures)}
        self.options_by_day = {service.valuation_day: tuple(sorted(self.contracts))}

    def refresh(self):
        return self.service.refresh()

    def snapshot(self, now_ns=None):
        return self.service.snapshot(now_ns)

    @property
    def manifest(self):
        return self.service.manifest()


class OptionSessionRunner(UnifiedStrategyRunner):
    """从独立Feed收集收盘Bar，串行发布帧；不要求所有候选期权每分钟成交。"""
    def __init__(self, *, positions, references, sources, labels, replay, grace_seconds,
                 market_age_seconds, instrument_age_seconds, expected_day, clock_ns=None):
        super().__init__(RuntimeMode.LIVE, position_manager=positions)
        self.references, self.sources, self.labels = references, dict(sources), dict(labels)
        self.expected_day = expected_day
        self.clock_ns = clock_ns or time.time_ns
        self.replay = replay
        self.grace_seconds, self.market_age_ns = grace_seconds, int(market_age_seconds * 1e9)
        self.instrument_age_seconds = instrument_age_seconds
        self.frames = MinuteFrameBuffer(sources, grace_seconds=grace_seconds)
        self._frame_lock = threading.RLock()
        self.accept_bars = False
        self.warm_started_ns = None
        self.failure = None
        self.observations = {}
        self.readiness = None
        self.strategy = self.client = self.manager = self.prices = None
        self.ctp_upstream = None
        self.feature_query_feeds = ()
        self.skipped_feature_frames = 0
        self.frame_audit = []
        self.fixed_ids = frozenset(references.instrument_ids.values())

    def poll_features(self):
        if self.feature_query_feeds:
            with self._frame_lock:
                pending = tuple(self.frames.pending) if self.accept_bars else ()
            # 数据库IO在控制线程、帧锁之外；回调沿公共分钟缓冲判断期限。
            for feed in self.feature_query_feeds:
                feed.poll(pending)

    def reference_is_current(self):
        return self.references.snapshot().fingerprint == self.references.fingerprint

    def session_ready(self):
        return not self.failure and self.reference_is_current() and bool(self.readiness())

    def begin_bars(self):
        with self._frame_lock:
            self.frames = MinuteFrameBuffer(self.sources, grace_seconds=self.grace_seconds)
            self.warm_started_ns = self.clock_ns()
            self.accept_bars = True

    def close_role_gate(self, reason):
        self.accept_bars = False
        self.failure = reason
        raise RuntimeError(reason)

    def publish(self, feed_id, event):
        if not isinstance(event, Bar):
            return
        with self._frame_lock:
            try:
                key = self.key_by_instrument.get(event.bar_type.instrument_id)
                if key is None or self.sources[key] != feed_id:
                    raise ValueError("行情合约或来源不属于本次绑定")
                now = self.clock_ns()
                valuation = minute_close_ns(event.ts_event, self.labels[feed_id])
                day = datetime.fromtimestamp(valuation / 1e9, ZoneInfo("Asia/Shanghai")).date()
                if day != self.references.service.valuation_day:
                    raise ValueError(f"行情估值日期不一致: {feed_id}/{key}/{day}")
                if not self.replay and not 0 <= now - valuation <= self.market_age_ns:
                    raise ValueError("实时Bar过期或尚未收盘")
                previous = self.observations.get(key)
                if previous and valuation < previous[0]:
                    raise ValueError(f"单合约行情时间倒退: {key}")
                if self.accept_bars:
                    if not self.frames.add(key, event.close.as_decimal(), valuation, now,
                            source_id=feed_id, source_ns=event.ts_event):
                        return
                self.observations[key] = (valuation, now)
            except Exception as error:
                self.failure = str(error)
                self.accept_bars = False
                raise

    def stale_instruments(self):
        now = self.clock_ns()
        required = self.strategy.required_market_keys()
        warming = self.warm_started_ns is not None and 0 <= now - self.warm_started_ns <= self.market_age_ns
        stale = {key: {"last": self.observations.get(key)} for key in required
            if (key not in self.observations and not warming)
            or (key in self.observations and not 0 <= now - self.observations[key][1] <= self.market_age_ns)}
        if self.ctp_upstream is not None:
            ids = [self.references.instrument_ids[key] for key in required if key in self.references.instrument_ids]
            health = InstrumentHealthGate(self.ctp_upstream, ids, self.expected_day,
                policy=self.ctp_profile, max_age_seconds=self.instrument_age_seconds)
            raw = health.stale_instruments()
            stale.update({key: value for key, value in raw.items()
                if not warming or value["trading_day"] is not None})
        return stale

    def flush_frames(self):
        with self._frame_lock:
            if not self.accept_bars:
                return
            try:
                # 控制线程已刷新参考；帧锁内仅检查快照，避免数据库读取阻塞行情。
                if not self.session_ready():
                    raise RuntimeError("分钟决策前参考或行情通道未就绪")
                for frame in self.frames.close_due(self.clock_ns()):
                    audit = {"valuation_ns": frame.valuation_ns, "request_ns": frame.request_ns,
                        "deadline_ns": min(item.received_ns for item in frame.observations.values()) + self.frames.grace_ns,
                        "observations": {key: {"source_id": item.source_id,
                            "source_ns": item.source_ns, "received_ns": item.received_ns}
                            for key, item in frame.observations.items()}, "status": "DELIVERED"}
                    self.frame_audit.append(audit)
                    del self.frame_audit[:-256]
                    if any(feed.key not in frame.observations for feed in self.feature_query_feeds):
                        # DFS指数通道显式要求同分钟输入；期限后缺失则跳过整次决策。
                        self.skipped_feature_frames += 1
                        audit["status"] = "SKIPPED_INDEX"
                        continue
                    # 每个执行价使用实际接收时刻；策略缓存只加入当前封闭帧。
                    for key, observation in frame.observations.items():
                        instrument = self.references.instrument_ids.get(key)
                        if instrument is not None:
                            self.prices.update(instrument, observation.price, observation.received_ns)
                    self.strategy.on_market_frame(frame)
                    audit["status"] = "CONSUMED"
                    if self.manager is not None:
                        self.manager.save()
            except Exception as error:
                self.failure = str(error)
                self.accept_bars = False
                raise


def assemble(args, inputs, driver=None, *, account_routes=()):
    refs, feeds = inputs.references, inputs.feeds
    ids = refs.instrument_ids
    positions = PositionManager()
    runner = OptionSessionRunner(positions=positions, references=refs, sources=inputs.sources,
        labels=inputs.labels, replay=args.simnow_environment == "replay",
        grace_seconds=args.frame_wait_seconds, market_age_seconds=args.max_market_age_seconds,
        instrument_age_seconds=args.instrument_max_age_seconds, expected_day=args.expected_trading_day)
    runner.key_by_instrument = {value: key for key, value in ids.items()}
    runner.key_by_instrument[inputs.index_id] = args.index_code
    multipliers = {ids[key]: info.multiplier for key, info in refs.contracts.items()}
    multipliers.update({ids[key]: row["contMultNum"] for key, row in refs.futures.items()})
    limits = risk_limits(args.config, refs.contracts, ids, multipliers, args.max_notional)
    scope = InstrumentPositionScope(ids.values())
    policy = execution_policy_from_args(args)
    if args.mode == "model":
        execution = build_model_execution(CLIENT_ID, instrument_limits=limits, positions=positions)
    elif args.mode == "recording":
        execution = ExecutionComponents(RecordingExecutionClient(CLIENT_ID), positions, MarketReferencePriceStore())
    else:
        execution = build_ctp_portfolio_execution(driver, trading_day=args.expected_trading_day,
            price_increments={ids[key]: info.tick for key, info in refs.contracts.items()} |
                {ids[key]: row["minChgPriceNum"] for key, row in refs.futures.items()},
            multipliers=multipliers, instrument_limits=limits, session_check=runner.session_ready,
            orders=True, positions=positions, limit_offset_ticks=args.limit_offset_ticks,
            instrument_scope=scope, account_reconciliation=policy.account_reconciliation,
            generic_close_venues=("CFFEX",))
        from .execution_audit import AuditedExecutionBackend
        audit = AuditedExecutionBackend(execution.backend, execution.prices,
            max_quote_age_seconds=args.instrument_max_age_seconds)
        execution.client.backend = audit
        execution = replace(execution, backend=audit)
        if inputs.ctp_upstream is not None:
            from bomber.framework.market.basic.base import DataType
            inputs.ctp_upstream.register_quote_tick_handler(audit.observe_quote)
            for instrument in ids.values():
                inputs.ctp_upstream.subscribe(instrument, DataType.QUOTE_TICK)
    if account_routes:
        if args.mode != "model":
            raise ValueError("跟随端须通过独立策略主账本装配")
        execution = build_fanout_execution(execution, account_routes)
    if args.mode == "recording":
        from bomber.framework.trader.execution.policy import AccountReconciliationMode
        execution.client.account_reconciliation = AccountReconciliationMode.OFF
    runner.client, runner.prices = execution.client, execution.prices
    channels = []
    if inputs.ctp_upstream is not None:
        runner.ctp_upstream = inputs.ctp_upstream
        runner.ctp_profile = CtpLiveProfile(args.simnow_environment, args.replay_md_trading_day)
        channels.append(CtpMarketChannel(inputs.ctp_upstream, args.expected_trading_day,
            profile=runner.ctp_profile, max_age_seconds=args.instrument_max_age_seconds))
    channels.extend(DolphinMarketChannel(feed, max_receive_age_seconds=args.max_market_age_seconds,
        allow_historical=runner.replay) for feed in inputs.dolphin_feeds)
    runner.feature_query_feeds = tuple(getattr(inputs, "feature_query_feeds", ()))
    if runner.feature_query_feeds:
        from bomber.framework.trader.runtime.live.channels.features import FeatureQueryChannel
        channels.extend(FeatureQueryChannel(feed.cache) for feed in runner.feature_query_feeds)
    channels.append(CtpExecutionChannel(driver, args.expected_trading_day, require_orders=True) if driver else
        LocalExecutionChannel(CLIENT_ID, lambda: bool(getattr(execution.client, "is_reconciled",
            getattr(execution.client, "is_started", False))),
            channel="recording" if args.mode == "recording" else "model"))
    readiness = runner.readiness = SessionReadiness(*channels)
    strategy = assemble_strategy_components(runner, execution, config=args.config,
        contracts=refs.contracts, futures=refs.futures, instruments=ids,
        index_key=args.index_code, index_id=inputs.index_id, trading_days=refs.calendar,
        options_by_day=refs.options_by_day, feeds=feeds, feed_by_key=inputs.sources,
        final_day=args.flatten_day, strategy_id=CLIENT_ID)
    runner.strategy = strategy
    binding = StrategyPositionBinding(CLIENT_ID, execution.client.client_id, {str(i): i for i in ids.values()})
    session = SimpleNamespace(mode=args.mode, runner=runner, strategy=strategy, client=execution.client,
        driver=driver, ledger=execution.ledger, references=refs, readiness=readiness,
        recovery_binding=binding, position_scope=scope, upstream=inputs.ctp_upstream,
        bar_feed=inputs.ctp_bar_feed, manager=None, feeds=feeds, execution=execution,
        source_identity=inputs.source_identity)
    if args.state_file is not None:
        import json
        identity = {"strategy_id": CLIENT_ID, "config": vars(args.config),
            "valuation_day": str(refs.service.valuation_day), "td_day": args.expected_trading_day,
            "reference_fingerprint": refs.fingerprint, "calendar": tuple(map(str, refs.calendar)),
            "calendar_manifest": refs.calendar_manifest,
            "contracts": {key: vars(info) for key, info in refs.contracts.items()},
            "sources": inputs.source_identity, "labels": inputs.labels, "mode": args.mode,
            "environment": args.simnow_environment, "md_day": args.replay_md_trading_day,
            "flatten_day": args.flatten_day, "policy": policy.snapshot_state(),
            "max_notional": args.max_notional, "frame_wait_seconds": args.frame_wait_seconds,
            "limit_offset_ticks": args.limit_offset_ticks, "ctp_price_type": args.ctp_price_type}
        components = {"session_identity": SessionIdentityCheckpoint(json.loads(json.dumps(identity, default=str))),
                      "vega_signal": OptionVegaCheckpoint(strategy)}
        attach = attach_ctp_persistence if args.mode == "simnow" else attach_execution_persistence
        session.manager = attach(execution, runner, state_file=args.state_file, state_components=components)
        runner.manager = session.manager
    return session
