"""CTP渠道独立适配：MD/TD检查、SimNow配置及原有受控会话生命周期。"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
import time
from urllib.parse import urlsplit

from ..contracts import (LiveRunContext, ChannelIdentity, ReadinessIssue, ReadinessResult, SessionReadiness)
from ..profiles import MarketTimeProfile
from ..health import validate_instrument_max_age, timestamp_issue
from ..recovery import InstrumentPositionScope


def inspect_adopted_position(*args, **kwargs):
    from ...ctp_positions import inspect_adopted_position as inspect
    return inspect(*args, **kwargs)


@dataclass(frozen=True)
class CtpLiveProfile(MarketTimeProfile):
    environment: str = "realtime"
    replay_md_trading_day: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.environment == "replay":
            value = self.replay_md_trading_day or ""
            if (len(value) != 8 or not value.isdigit() or
                    datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") != value):
                raise ValueError("replay须提供有效的--replay-md-trading-day YYYYMMDD")
        elif self.replay_md_trading_day:
            raise ValueError("--replay-md-trading-day仅用于replay模式")

    def expected_md_day(self, trading_day):
        return self.replay_md_trading_day or trading_day

LiveTimePolicy = CtpLiveProfile  # 旧入口名称兼容


def validate_replay_environment(md_front, td_front, production_mode):
    for front, port in ((md_front, 40011), (td_front, 40001)):
        parsed = urlsplit(front)
        if (parsed.scheme != "tcp" or parsed.hostname != "182.254.243.31" or parsed.port != port
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("replay模式须使用同组SimNow MD40011／TD40001前置")
    if not production_mode:
        raise ValueError("本次第二套看穿式前置须CTP_PRODUCTION_MODE=true")


def add_environment_arguments(parser):
    parser.add_argument("--simnow-environment", choices=("realtime", "replay"), default="realtime",
                        help="realtime检查事件时效；replay按接收时间聚合第二套SimNow行情")
    parser.add_argument("--replay-md-trading-day", help="replay必填：原始行情TradingDay，YYYYMMDD")
    parser.add_argument("--instrument-max-age-seconds", type=float, default=10,
                        help="逐合约时效(0,120]秒；realtime检查事件/接收，replay检查接收")


def validate_environment_arguments(args):
    policy = CtpLiveProfile(args.simnow_environment, args.replay_md_trading_day)
    validate_instrument_max_age(args.instrument_max_age_seconds)
    return policy


class InstrumentHealthGate:
    """原始深度必须覆盖每个固定合约；回放只豁免历史事件年龄上限。

    fallback仅供没有深度接口的手动Feed使用。真实CTP的空深度字典仍拒绝。
    """
    def __init__(self, upstream, instruments, expected_day, *, policy=None,
                 max_age_seconds=10, clock_ns=None, monotonic_ns=None,
                 fallback_events=None, fallback_receives=None):
        self.upstream = upstream
        self.instruments = frozenset(instruments)
        self.policy = policy or CtpLiveProfile()
        self.expected_day = self.policy.expected_md_day(expected_day)
        self.max_age_ns = validate_instrument_max_age(max_age_seconds)
        self.clock_ns = clock_ns or (lambda: time.time_ns())
        self.monotonic_ns = monotonic_ns or (lambda: time.monotonic_ns())
        self.fallback_events = fallback_events
        self.fallback_receives = fallback_receives
        self.depth_records = {}

    def stale_instruments(self):
        now, received = self.clock_ns(), self.monotonic_ns()
        observations = getattr(self.upstream, "depth_observations", None)
        if observations is None and self.fallback_events is None:
            return {}  # 非CTP Feed继续由自身健康接口门控。
        if observations is not None:
            self.depth_records = dict(observations)
        failures = {}
        identity = ChannelIdentity("market", "ctp", getattr(self.upstream, "source_id", "ctp-md"))
        for instrument in self.instruments:
            observation = observations.get(instrument) if observations is not None else None
            event = (observation.ts_event if observation else
                     self.fallback_events.get(instrument) if observations is None else None)
            receive = (observation.received_monotonic_ns if observation else
                       (self.fallback_receives or {}).get(instrument) if observations is None else None)
            day_ok = observations is None or (observation and observation.trading_day == self.expected_day)
            regressed = bool(observation and observation.timestamp_regressed)
            event_age = None if event is None else now - event
            receive_age = None if receive is None else received - receive
            event_issue = timestamp_issue(identity, "EVENT", event, now, self.max_age_ns,
                                          allow_historical=self.policy.allow_historical_events)
            receive_issue = timestamp_issue(identity, "RECEIVE", receive, received, self.max_age_ns)
            if not day_ok or regressed or event_issue is not None or receive_issue is not None:
                failures[str(instrument)] = {
                    "event_age_seconds": None if event_age is None else event_age / 1e9,
                    "receive_age_seconds": None if receive_age is None else receive_age / 1e9,
                    "trading_day": observation.trading_day if observation else None,
                    "max_age_seconds": self.max_age_ns / 1e9, "timestamp_regressed": regressed}
        return failures


def required(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"缺少环境变量: {name}")
    return value


def build_simnow_transport(transport_factory, *, client_id, flow_path, timeout_seconds):
    if required("CTP_BROKER_ID") != "9999":
        raise ValueError("本入口仅允许SimNow BrokerID=9999")
    investor = required("CTP_ACCOUNT_ID")
    return transport_factory(client_id=client_id, account_id=investor,
        front=required("CTP_TD_ADDRESS"), broker_id="9999", investor_id=investor,
        password=required("CTP_PASSWORD"), app_id=os.getenv("CTP_APP_ID", ""),
        auth_code=os.getenv("CTP_AUTH_CODE", ""), flow_path=os.getenv("CTP_TD_FLOW_PATH", flow_path),
        production_mode=os.getenv("CTP_PRODUCTION_MODE", "true").lower() in {"1", "true", "yes", "on"},
        timeout_seconds=timeout_seconds)


def build_md_config(config_factory, lifecycle, *, flow_path):
    credentials = getattr(lifecycle, "md_credentials", None)
    if credentials is not None:
        return config_factory(front=lifecycle.md_front, broker_id=credentials.broker_id,
            user_id=credentials.user_id, password=credentials.password,
            flow_path=os.getenv("CTP_MD_FLOW_PATH", flow_path), production_mode=credentials.production_mode)
    return config_factory(front=lifecycle.md_front, broker_id="9999",
        user_id=lifecycle.transport.investor_id, password=lifecycle.transport.password,
        flow_path=os.getenv("CTP_MD_FLOW_PATH", flow_path),
        production_mode=lifecycle.transport.production_mode)

def occupied_positions(transport):
    return {str(key): value for key, value in transport.query_gross_positions().items() if any(value)}


def assert_account_ready(driver, *, reconcile_positions=True):
    if reconcile_positions:
        driver.reconcile()
    account = driver.reconcile_account_state()
    if driver.reconcile_active_orders().orders:
        raise RuntimeError("账户已有活动订单，拒绝自动接管")
    if "CNY" not in account.balances or account.balances["CNY"].available <= 0:
        raise RuntimeError("权威查询未取得可用CNY资金")
    return account


def assert_flat_account(driver, transport, *, position_scope=None):
    account = assert_account_ready(driver)
    gross = occupied_positions(transport)
    managed = gross if position_scope is None else position_scope.select(gross)
    if managed:
        raise RuntimeError(f"账户已有总仓，拒绝空仓启动: {managed}")
    return account


class CtpMarketChannel:
    """CTP MD独立适配：健康、接收心跳、MD交易日和逐合约原始深度。"""
    def __init__(self, upstream, expected_day, *, profile=None, instrument_health=None,
                 max_age_seconds=10, monotonic_ns=None, source_id=None):
        self.upstream = upstream
        self.profile = profile or CtpLiveProfile()
        self.expected_day = self.profile.expected_md_day(expected_day)
        self.instrument_health = instrument_health
        self.max_age_ns = validate_instrument_max_age(max_age_seconds)
        self.monotonic_ns = monotonic_ns or (lambda: time.monotonic_ns())
        self.identity = ChannelIdentity("market", "ctp", source_id or getattr(upstream, "source_id", "ctp-md"))

    def check(self):
        issues = []
        health = self.upstream.health_snapshot
        if getattr(health.state, "value", health.state) != "ready":
            issues.append(ReadinessIssue(self.identity, "STREAM_UNHEALTHY",
                str(getattr(health, "reason", health.state))))
        if self.upstream.latest_trading_day != self.expected_day:
            issues.append(ReadinessIssue(self.identity, "MD_TRADING_DAY_MISMATCH", "MD交易日不符",
                {"actual": self.upstream.latest_trading_day, "expected": self.expected_day}))
        issue = timestamp_issue(self.identity, "RECEIVE", self.upstream.latest_receive_monotonic_ns,
                                self.monotonic_ns(), self.max_age_ns)
        if issue is not None:
            issues.append(issue)
        if self.instrument_health is not None:
            for instrument, details in self.instrument_health.stale_instruments().items():
                issues.append(ReadinessIssue(self.identity, "INSTRUMENT_UNHEALTHY", instrument, details))
        return ReadinessResult((self.identity,), tuple(issues))


class CtpExecutionChannel:
    """CTP TD独立适配；不读取行情源、MD日期或分钟Bar。"""
    def __init__(self, driver, expected_day, *, require_orders=False, source_id=None,
                 connection_ready=None):
        self.driver, self.expected_day = driver, expected_day
        self.require_orders = require_orders
        # 原生Driver在断线后仍保留TradingDay；不能仅以日期证明连接有效。
        # 第三方Driver可以显式注入自己的连接状态接口。
        self.connection_ready = connection_ready or (
            lambda: getattr(driver, "_connected", driver.trading_day is not None))
        self.identity = ChannelIdentity("execution", "ctp", source_id or getattr(driver, "driver_id", "ctp-td"))

    def check(self):
        issues = []
        if not self.connection_ready():
            issues.append(ReadinessIssue(self.identity, "EXECUTION_DISCONNECTED", "CTP交易连接已失效"))
        if self.driver.trading_day != self.expected_day:
            issues.append(ReadinessIssue(self.identity, "TD_TRADING_DAY_MISMATCH", "TD交易日不符",
                {"actual": self.driver.trading_day, "expected": self.expected_day}))
        if self.require_orders and not self.driver.is_simnow_session:
            issues.append(ReadinessIssue(self.identity, "EXECUTION_NOT_AUTHORIZED", "尚未取得SimNow交易会话"))
        return ReadinessResult((self.identity,), tuple(issues))


def build_session_readiness(driver, upstream, expected_day, *, profile=None,
                            require_orders=False, instrument_health=None, monotonic_ns=None):
    return SessionReadiness(
        CtpMarketChannel(upstream, expected_day, profile=profile,
                         instrument_health=instrument_health, monotonic_ns=monotonic_ns),
        CtpExecutionChannel(driver, expected_day, require_orders=require_orders))


def session_ready(driver, upstream, expected_day, *, require_orders=False, now_ns=None,
                  replay_md_trading_day=None):
    """旧布尔接口兼容；新装配可独立组合MD/TD，并读取结构化check结果。"""
    profile = CtpLiveProfile("replay" if replay_md_trading_day else "realtime", replay_md_trading_day)
    return bool(build_session_readiness(driver, upstream, expected_day,
        profile=profile, require_orders=require_orders,
        monotonic_ns=None if now_ns is None else lambda: now_ns).check())


def readiness_details(session):
    readiness = getattr(session, "readiness", None)
    result = getattr(readiness, "last_result", None)
    return result.as_dict() if result is not None else None


@contextmanager
def account_lock(account_id, *, namespace="bomber-ctp"):
    import fcntl
    key = hashlib.sha256(account_id.encode()).hexdigest()[:16]
    with open(f"/tmp/{namespace}-{key}.lock", "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("同一账户已有进程运行") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class CtpSessionLifecycle:
    """ManagedLiveRuntime控制器；不读取EMA属性，不实现LIVE动态换约。"""
    def __init__(self, transport, *, md_front, orders=False, max_session_orders=4,
                 environment="realtime", replay_md_trading_day=None, expected_trading_day=None,
                 adopt_instrument=None, expected_position=None,
                 scope_to_reference_instruments=False,
                 account_reconciliation="strict",
                 lock_namespace="bomber-ctp", legacy_lock_namespaces=(),
                 ready_seconds=30, reconcile_seconds=10):
        if transport.broker_id != "9999":
            raise ValueError("本生命周期仅允许SimNow BrokerID=9999")
        self.profile = CtpLiveProfile(environment, replay_md_trading_day)
        self.account_reconciliation = getattr(account_reconciliation, "value", account_reconciliation)
        if self.account_reconciliation not in {"strict", "observe", "off"}:
            raise ValueError("未知账户对账政策")
        self.position_differences = {}
        if ready_seconds <= 0 or reconcile_seconds <= 0 or max_session_orders <= 0:
            raise ValueError("就绪超时、对账间隔及会话订单上限须为正数")
        if environment == "replay":
            validate_replay_environment(md_front, transport.front, transport.production_mode)
        if (adopt_instrument is None) != (expected_position is None):
            raise ValueError("接管须同时声明合约及预期仓位")
        if adopt_instrument is not None and orders and max_session_orders < 2:
            raise ValueError("接管报单上限至少2笔以允许先平后开")
        self.transport = transport
        self.md_front = md_front
        self.orders = orders
        self.max_session_orders = max_session_orders
        self.environment = environment
        self.replay_md_trading_day = replay_md_trading_day
        self.expected_trading_day = expected_trading_day
        self.adopt_instrument = adopt_instrument
        self.expected_position = expected_position
        self.adopted_position = None
        if scope_to_reference_instruments and adopt_instrument is not None:
            raise ValueError("合约范围启动与单合约接管不能混用")
        if self.account_reconciliation != "strict" and adopt_instrument is not None:
            raise ValueError("独立成交账本不能自动接管柜台仓位")
        self.scope_to_reference_instruments = scope_to_reference_instruments
        self.position_scope = None
        self.external_gross = {}
        self.lock_namespace = lock_namespace
        self.legacy_lock_namespaces = tuple(legacy_lock_namespaces)
        self.ready_seconds = ready_seconds
        self.reconcile_seconds = reconcile_seconds
        self.lost = []
        self.session = None
        self._next_check = 0
        self._cap_deadline = None
        self._observed_md_day = None
        self._observed_td_day = None
        from bomber.framework.trader.execution.ctp import CtpNativeTraderDriver
        self.driver = CtpNativeTraderDriver(transport.client_id, transport.account_id, transport,
            enable_simnow_orders=orders, max_session_orders=max_session_orders,
            disconnect_handler=self._disconnected)

    @property
    def event_label(self):
        return "执行回报" if self.orders else "Recording目标"

    def _disconnected(self, reason):
        self.lost.append(reason)
        if self.session is not None and self.orders:
            self.session.client.mark_disconnected(reason)

    def prepare(self, resources):
        for namespace in dict.fromkeys((self.lock_namespace, *self.legacy_lock_namespaces)):
            resources.enter_context(account_lock(self.transport.account_id, namespace=namespace))
        self.driver.start(lambda report: None)
        if self.account_reconciliation != "strict":
            assert_account_ready(self.driver, reconcile_positions=False)
        elif self.adopt_instrument is None:
            if self.scope_to_reference_instruments:
                # 合约范围由本次权威参考资料确定；此阶段尚未装配或授权。
                assert_account_ready(self.driver)
            else:
                assert_flat_account(self.driver, self.transport)
        else:
            self.adopted_position = inspect_adopted_position(self.driver, self.transport,
                self.adopt_instrument, self.expected_position)
        day = self.driver.trading_day
        self._observed_td_day = day
        if self.expected_trading_day and self.expected_trading_day != day:
            raise RuntimeError("TD交易日与指定交易日不一致")
        print(f"本次会话: environment={self.environment} TD交易日={day} "
              f"MD要求交易日={self.replay_md_trading_day or day} MD={self.md_front} "
              f"TD={self.transport.front} bar_time_basis={getattr(self, 'bar_time_basis', self.profile.bar_time_basis)}",
              flush=True)
        return LiveRunContext(day, resources)

    def finish_preparation(self, context):
        if self.scope_to_reference_instruments:
            self.position_scope = InstrumentPositionScope(context.references.instrument_ids.values())
            if not getattr(self, "resuming", False) and self.account_reconciliation == "strict":
                assert_flat_account(self.driver, self.transport, position_scope=self.position_scope)
            if self.account_reconciliation != "off":
                self.external_gross = self.position_scope.external(occupied_positions(self.transport))
            print(f"策略管理合约: {sorted(self.position_scope.instruments)} "
                  f"范围外柜台总仓: {self.external_gross}", flush=True)
        if self.adopted_position is not None and context.references.instrument_id != self.adopted_position.instrument:
            raise RuntimeError("待接管合约不是本次参考资料主力，拒绝接管")
        # 参考/行情准备完毕后结束只读预检；下一步安装最终会话持久化钩子。
        self.driver.stop()

    def start(self, session, context):
        self.session = session
        self._validate_position_scope(session)
        if self.orders:
            session.client.start()
        else:
            self.driver.start(lambda report: None)
        if self.account_reconciliation != "strict":
            assert_account_ready(self.driver, reconcile_positions=False)
        elif self.adopted_position is None:
            assert_flat_account(self.driver, self.transport, position_scope=self.position_scope)
        else:
            current = inspect_adopted_position(self.driver, self.transport,
                self.adopt_instrument, self.expected_position)
            if current != self.adopted_position:
                raise RuntimeError("重新登录后接管仓位与预检不同")
            if not self.orders:
                session.runner.position_manager.replace_account_positions(self.driver.driver_id,
                    {current.instrument: current.quantity})
            session.runner.adopt_initial_position(current.quantity)
            session.ledger.restore(current.ledger_state(session.references.spec.multiplier))
            print(f"已核验接管仓位: {current.instrument} net={current.quantity} "
                  f"PositionDate={current.position_date} PositionCost={current.position_cost}", flush=True)
        if self.driver.trading_day != context.trading_day or self.lost:
            raise RuntimeError("装配期间TD会话／交易日发生变化")
        session.runner.start()
        deadline = time.monotonic() + self.ready_seconds
        while not session.runner.session_ready():
            if self.lost or time.monotonic() >= deadline:
                raise RuntimeError(f"{self.ready_seconds:g}秒内未取得要求的健康MD／TD会话："
                    f"TD={self.driver.trading_day} MD={session.upstream.latest_trading_day} "
                    f"要求MD={self.replay_md_trading_day or context.trading_day} "
                    f"health={session.upstream.health_snapshot.state} "
                    f"readiness={readiness_details(session)}")
            time.sleep(0.2)
        prepare_history = getattr(session.runner, "prepare_history", None)
        if callable(prepare_history):
            session.runner.mark_history_live_start()
            prepare_history()
            if not session.runner.session_ready():
                raise RuntimeError("历史加载后MD/TD未就绪，保持闭闸")
        if self.orders:
            session.manager.save()
            session.client.arm_demo(session.client.DEMO_CONFIRMATION)
        session.runner.begin_bars()
        self._next_check = time.monotonic() + self.reconcile_seconds
        print(f"会话就绪，开始分钟验证：{self.environment}；停机撤单并报告剩余仓位", flush=True)

    def poll(self, session):
        if session.runner.failure or self.lost:
            raise RuntimeError(session.runner.failure or self.lost[-1])
        if not session.runner.session_ready():
            upstream = session.upstream
            received = upstream.latest_receive_monotonic_ns
            age = None if received is None else (time.monotonic_ns() - received) / 1_000_000_000
            health = upstream.health_snapshot
            raise RuntimeError("MD／TD交易日变化、行情异常或停滞，停止本次运行："
                f"TD={self.driver.trading_day} expected_TD={session.runner.expected_day} "
                f"MD={upstream.latest_trading_day} expected_MD={self.replay_md_trading_day or session.runner.expected_day} "
                f"health={health.state} reason={getattr(health, 'reason', None)} "
                f"receive_age_seconds={age} max_receive_age_seconds=10 "
                f"simnow_session={self.driver.is_simnow_session} "
                f"readiness={readiness_details(session)}")
        if not getattr(session.references, "refresh_on_event", True):
            session.references.refresh()
        session.references.snapshot(time.time_ns())
        reference_check = getattr(session.runner, "reference_is_current", None)
        if (not reference_check() if callable(reference_check) else
                session.references.spec != session.runner.fixed_spec):
            session.runner.close_role_gate("角色／条款变更，闭闸重新核对")
        if self.orders:
            if session.client.report_errors:
                raise RuntimeError("执行回报冲突，停止并要求对账")
            if self.driver.submitted_orders >= self.max_session_orders and self._cap_deadline is None:
                session.runner.accept_bars = False
                session.client.disarm("session_order_cap")
                self._cap_deadline = time.monotonic() + 10
            if self._cap_deadline is not None and time.monotonic() >= self._cap_deadline:
                return False
        if time.monotonic() >= self._next_check:
            if self.orders:
                session.client.refresh_account_state()
            else:
                self.driver.reconcile_account_state()
            if self.account_reconciliation != "off":
                gross = occupied_positions(self.transport)
                actual = self._managed_gross(gross)
                expected = self._expected_gross(session)
                self.position_differences = {i: {"observed": actual.get(i, (0, 0)), "execution": expected.get(i, (0, 0))}
                    for i in actual.keys() | expected.keys() if actual.get(i, (0, 0)) != expected.get(i, (0, 0))}
                if self.position_differences and self.account_reconciliation == "strict":
                    raise RuntimeError(f"柜台总仓与本地账本不一致: actual={actual} expected={expected}")
            self._next_check = time.monotonic() + self.reconcile_seconds
        return True

    def _validate_position_scope(self, session):
        if self.scope_to_reference_instruments and (
                self.position_scope is None or getattr(session, "position_scope", None) != self.position_scope):
            raise RuntimeError("执行持仓范围未绑定或与参考合约不一致")

    def _managed_gross(self, gross):
        if self.position_scope is None:
            return gross
        self.external_gross = self.position_scope.external(gross)
        return self.position_scope.select(gross)

    @staticmethod
    def _expected_gross(session):
        instrument = session.references.instrument_id
        snap = session.ledger.snapshot(instrument)
        return {} if not (snap.long_total or snap.short_total) else {
            str(instrument): (snap.long_total, snap.short_total)}

    def verify(self, session):
        from bomber.framework.trader.execution.contracts import ExecutionReportType
        if not self.orders and not session.client.requests:
            raise RuntimeError("未取得当前策略Recording目标；检查行情时段、资料和预热时长")
        if self.orders and not any(r.report_type in {
            ExecutionReportType.PARTIALLY_FILLED, ExecutionReportType.FILLED}
                for r in session.client.backend.reports):
            raise RuntimeError("本次没有成交回报，不能判定SimNow交易通过")

    def events(self, session):
        if session is None:
            return ()
        return session.client.backend.reports if self.orders else session.client.requests

    def shutdown(self, session, context):
        errors = []
        final_gross = final_active = None
        self._observed_md_day = getattr(session.upstream, "latest_trading_day", None) if session else None
        self._observed_td_day = self.driver.trading_day or self._observed_td_day
        if session:
            session.runner.accept_bars = False
            if self.orders:
                try:
                    session.client.set_risk_mode("HALTED", cancel_active_orders=True)
                    until = time.monotonic() + 10
                    while self.driver.reconcile_active_orders().orders:
                        if time.monotonic() >= until:
                            raise RuntimeError("停机仍有活动订单，需要人工核对")
                        time.sleep(1)
                except Exception as error:
                    errors.append(str(error))
        try:
            # 预检已结束但组装失败时尚无最终会话，也没有可查询的连接。
            if session is not None or self.driver.trading_day is not None:
                final_active = len(self.driver.reconcile_active_orders().orders)
                if self.account_reconciliation != "off":
                    final_gross = occupied_positions(self.transport)
                    self._managed_gross(final_gross)
                self.driver.reconcile_account_state()
            if session and self.orders:
                session.manager.save()
                if self.account_reconciliation == "strict" and self._managed_gross(final_gross) != self._expected_gross(session):
                    raise RuntimeError("停机柜台总仓与本地账本不一致")
        except Exception as error:
            errors.append(str(error))
        try:
            if session:
                session.runner.stop()
        except Exception as error:
            errors.append(str(error))
        finally:
            try:
                self.driver.stop()
            except Exception as error:
                errors.append(str(error))
        result = {"final_gross": final_gross, "final_active_orders": final_active, "cleanup_errors": errors}
        if self.position_scope is not None:
            result.update(final_managed_gross=None if final_gross is None else self.position_scope.select(final_gross),
                          external_gross=dict(self.external_gross))
        return result

    def snapshot(self, session, context):
        references = context.references if context else None
        return {
            "account_reconciliation": self.account_reconciliation,
            "strategy_position_source": "fills", "position_differences": self.position_differences,
            "managed_instruments": None if self.position_scope is None else sorted(self.position_scope.instruments),
            "external_gross": dict(self.external_gross),
            "simnow_environment": self.environment, "replay_md_trading_day": self.replay_md_trading_day,
            "observed_md_trading_day": self._observed_md_day, "observed_td_trading_day": self._observed_td_day,
            "bar_time_basis": self.profile.bar_time_basis,
            "reference_time_basis": "current_td_and_wall_clock",
            "timing": getattr(session.bar_feed, "timing_snapshot", {}) if session else {},
            "orders_submitted": self.driver.submitted_orders,
            "execution_audit": [
                {"ts_ns": item.ts_ns, "action": item.action, "detail": item.detail}
                for item in (getattr(session.client, "audit_records", ())[-30:] if session else ())
            ],
            "adopted_position": (None if self.adopted_position is None else {
                "instrument": str(self.adopted_position.instrument),
                "quantity": str(self.adopted_position.quantity),
                "position_date": self.adopted_position.position_date,
                "position_cost": str(self.adopted_position.position_cost),
                "trading_day": self.adopted_position.trading_day}),
            "trading_day": session.runner.expected_day if session else None,
            "reference_files": [str(p) for p in references.paths] if references else [],
            "reference_manifest": getattr(references, "manifest", {}),
            "fixed_instrument": str(session.runner.bound_instrument) if session else None,
            "factor_date": str(references.factor_date) if references else None,
            **({"factor_date_basis": references.factor_date_basis} if references else {}),
        }
