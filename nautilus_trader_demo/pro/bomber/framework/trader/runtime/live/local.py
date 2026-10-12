"""独立策略执行的Live生命周期；不登录、查询或锁定任何物理账户。"""

import time
from contextlib import contextmanager
from .contracts import ChannelIdentity, LiveRunContext, ReadinessResult
from .recovery import validate_finished_orders, validate_owned_positions, restored_target_revision


@contextmanager
def strategy_state_lock(state_file):
    """仅保护策略状态文件；不意味着占用或关联物理交易账户。"""
    import fcntl
    import hashlib
    from pathlib import Path
    key = hashlib.sha256(str(Path(state_file).resolve()).encode()).hexdigest()
    with open(f"/tmp/bomber-strategy-state-{key}.lock", "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("本策略状态文件已有进程使用") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class LocalExecutionChannel:
    def __init__(self, client_id, ready, *, channel="model"):
        self.identity = ChannelIdentity("execution", channel, client_id)
        self._ready = ready

    def check(self):
        from .contracts import ReadinessIssue
        issues = () if self._ready() else (ReadinessIssue(self.identity, "EXECUTION_NOT_READY", "策略执行未就绪"),)
        return ReadinessResult((self.identity,), issues)


class LocalExecutionLifecycle:
    event_label = "模型执行回报"

    def __init__(self, trading_day, *, resume=False, ready_seconds=30, state_file=None):
        self.trading_day = trading_day
        self.resuming = resume
        self.ready_seconds = ready_seconds
        self.resume_ready = False
        self.driver = None
        self.state_file = state_file

    def prepare(self, resources):
        if self.state_file is not None:
            resources.enter_context(strategy_state_lock(self.state_file))
        return LiveRunContext(self.trading_day, resources)

    def finish_preparation(self, context):
        pass

    def start(self, session, context):
        session.runner.accept_bars = False
        if self.resuming:
            if session.manager.restore() is None:
                raise RuntimeError("策略恢复文件不存在")
            validate_finished_orders(session)
            owned = validate_owned_positions(session.runner.position_manager, session.recovery_binding,
                                              account_scope=getattr(session, "position_scope", None))
            position_client = getattr(session.client, "position_client", session.client)
            risk = position_client.risk_manager
            for instrument, quantity in owned.items():
                limits = risk.instrument_limits[instrument]
                if limits.max_abs_position is not None and abs(quantity) > limits.max_abs_position:
                    raise RuntimeError("恢复策略仓位超过配置上限")
            revision = restored_target_revision(session, session.recovery_binding, allow_empty=True)
            session.strategy.restore_target_revision(revision)
            hold = getattr(session.runner, "hold_restored_target", None)
            if callable(hold):
                hold(session.strategy.strategy_id, revision)
        session.runner.start()
        deadline = time.monotonic() + self.ready_seconds
        while not session.runner.session_ready():
            if getattr(session.runner, "failure", None) or time.monotonic() >= deadline:
                raise RuntimeError("独立策略行情/执行未就绪")
            time.sleep(0.2)
        session.references.refresh()
        if not session.runner.session_ready():
            raise RuntimeError("独立策略参考或行情失效")
        if session.manager is not None:
            session.manager.save()
        self.resume_ready = True
        session.runner.begin_bars()
        print("策略会话就绪：独立参考价模型，无物理账户关联", flush=True)

    def poll(self, session):
        if getattr(session.runner, "failure", None) or session.client.report_errors:
            raise RuntimeError("独立策略执行或行情发生冲突")
        if not session.runner.session_ready():
            raise RuntimeError("独立策略行情交易日、健康或接收心跳失效")
        session.references.refresh()
        reference_check = getattr(session.runner, "reference_is_current", None)
        current = reference_check() if callable(reference_check) else session.references.spec == session.runner.fixed_spec
        if not current:
            session.runner.close_role_gate("独立策略参考合约或版本变化")
        stale = session.runner.stale_instruments()
        if stale:
            session.runner.close_role_gate(f"独立策略行情失效: {stale}")
        return True

    def verify(self, session):
        if session.client.report_errors or not any(getattr(r.report_type, "value", r.report_type) == "FILLED"
                                                 for r in session.client.backend.reports):
            raise RuntimeError("未取得策略模型成交，检查行情和信号预热")

    def events(self, session):
        return () if session is None else session.client.backend.reports

    def shutdown(self, session, context):
        if session is None:
            return {"cleanup_errors": []}
        session.runner.accept_bars = False
        try:
            if session.manager is not None and (not self.resuming or self.resume_ready):
                session.manager.save()
        finally:
            session.runner.stop()
        return {"cleanup_errors": [], "final_gross": None, "final_active_orders": None,
                "resume_state_preserved": self.resuming and not self.resume_ready,
                "model_reports": len(session.client.backend.reports)}

    def snapshot(self, session, context):
        return {"strategy_position_source": "model", "account_reconciliation": "off",
                "execution_origin": "model", "model": "reference-price",
                "trading_day": self.trading_day, "observed_td_trading_day": None,
                "observed_md_trading_day": None if session is None else session.upstream.latest_trading_day,
                "resume_ready": self.resume_ready,
                "account_dispatch": {} if session is None else getattr(session.client, "dispatch_status", {})}
