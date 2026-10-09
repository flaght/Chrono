"""单交易日、独占空仓CTP SimNow生命周期；默认只读，不含策略算法。"""

from contextlib import contextmanager
import hashlib
import time
from urllib.parse import urlsplit

from bomber.framework.market.stream.health import MarketHealthState
from bomber.framework.trader.execution.ctp import CtpNativeTraderDriver
from bomber.framework.trader.execution.contracts import ExecutionReportType
from .managed import LiveRunContext
from .ctp_positions import inspect_adopted_position


def occupied_positions(transport):
    return {str(key): value for key, value in transport.query_gross_positions().items() if any(value)}


def assert_flat_account(driver, transport):
    driver.reconcile()
    account = driver.reconcile_account_state()
    if driver.reconcile_active_orders().orders:
        raise RuntimeError("账户已有活动订单，拒绝自动接管")
    gross = occupied_positions(transport)
    if gross:
        raise RuntimeError(f"账户已有总仓，拒绝空仓启动: {gross}")
    if "CNY" not in account.balances or account.balances["CNY"].available <= 0:
        raise RuntimeError("权威查询未取得可用CNY资金")
    return account


def validate_replay_environment(md_front, td_front, production_mode):
    for front, port in ((md_front, 40011), (td_front, 40001)):
        parsed = urlsplit(front)
        if (parsed.scheme != "tcp" or parsed.hostname != "182.254.243.31" or parsed.port != port
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("replay模式须使用同组SimNow MD40011／TD40001前置")
    if not production_mode:
        raise ValueError("本次第二套看穿式前置须CTP_PRODUCTION_MODE=true")


def session_ready(driver, upstream, expected_day, *, require_orders=False, now_ns=None,
                  replay_md_trading_day=None):
    received = upstream.latest_receive_monotonic_ns
    age = None if received is None else (time.monotonic_ns() if now_ns is None else now_ns) - received
    return bool(driver.trading_day == expected_day
        and upstream.latest_trading_day == (replay_md_trading_day or expected_day)
        and upstream.health_snapshot.state is MarketHealthState.READY
        and age is not None and 0 <= age <= 10_000_000_000
        and (not require_orders or driver.is_simnow_session))


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
                 lock_namespace="bomber-ctp", legacy_lock_namespaces=(),
                 ready_seconds=30, reconcile_seconds=10):
        if transport.broker_id != "9999":
            raise ValueError("本生命周期仅允许SimNow BrokerID=9999")
        if environment not in {"realtime", "replay"}:
            raise ValueError("未知SimNow环境")
        if ready_seconds <= 0 or reconcile_seconds <= 0 or max_session_orders <= 0:
            raise ValueError("就绪超时、对账间隔及会话订单上限须为正数")
        if environment == "replay":
            if not replay_md_trading_day:
                raise ValueError("第二套联调缺少固定MD交易日")
            validate_replay_environment(md_front, transport.front, transport.production_mode)
        elif replay_md_trading_day:
            raise ValueError("原始回放交易日仅用于replay环境")
        if (adopt_instrument is None) != (expected_position is None):
            raise ValueError("接管须同时声明合约及预期仓位")
        if adopt_instrument is not None and (environment != "realtime" or (orders and max_session_orders < 2)):
            raise ValueError("接管须为实时会话；报单上限至少2笔以允许先平后开")
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
        if self.adopt_instrument is None:
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
              f"TD={self.transport.front} bar_time_basis={'receive' if self.replay_md_trading_day else 'event'}",
              flush=True)
        return LiveRunContext(day, resources)

    def finish_preparation(self, context):
        if self.adopted_position is not None and context.references.instrument_id != self.adopted_position.instrument:
            raise RuntimeError("待接管合约不是本次参考资料主力，拒绝接管")
        # 参考/行情准备完毕后结束只读预检；下一步安装最终会话持久化钩子。
        self.driver.stop()

    def start(self, session, context):
        self.session = session
        if self.orders:
            session.client.start()
        else:
            self.driver.start(lambda report: None)
        if self.adopted_position is None:
            assert_flat_account(self.driver, self.transport)
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
                    f"health={session.upstream.health_snapshot.state}")
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
                f"simnow_session={self.driver.is_simnow_session}")
        session.references.snapshot(time.time_ns())
        if session.references.spec != session.runner.fixed_spec:
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
            actual = occupied_positions(self.transport)
            expected = self._expected_gross(session)
            if actual != expected:
                raise RuntimeError(f"柜台总仓与本地账本不一致: actual={actual} expected={expected}")
            self._next_check = time.monotonic() + self.reconcile_seconds
        return True

    @staticmethod
    def _expected_gross(session):
        instrument = session.references.instrument_id
        snap = session.ledger.snapshot(instrument)
        return {} if not (snap.long_total or snap.short_total) else {
            str(instrument): (snap.long_total, snap.short_total)}

    def verify(self, session):
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
        self._observed_md_day = session.upstream.latest_trading_day if session else None
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
                final_gross = occupied_positions(self.transport)
                self.driver.reconcile_account_state()
            if session and self.orders:
                session.manager.save()
                if final_gross != self._expected_gross(session):
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
        return {"final_gross": final_gross, "final_active_orders": final_active, "cleanup_errors": errors}

    def snapshot(self, session, context):
        references = context.references if context else None
        return {
            "simnow_environment": self.environment, "replay_md_trading_day": self.replay_md_trading_day,
            "observed_md_trading_day": self._observed_md_day, "observed_td_trading_day": self._observed_td_day,
            "bar_time_basis": "receive" if self.replay_md_trading_day else "event",
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
