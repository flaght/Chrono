"""08自身的会话恢复与验收；通用账户、订单、归属检查使用公共模块。"""
import time

from bomber.framework.trader.runtime.live.local import LocalExecutionLifecycle, strategy_state_lock
from bomber.framework.trader.runtime.live.readiness import wait_for_session_ready
from bomber.framework.trader.runtime.live.recovery import (validate_finished_orders,
    validate_owned_positions, restored_target_revision)
from bomber.framework.trader.runtime.live.channels.ctp import (CtpSessionLifecycle,
    assert_flat_account, assert_account_ready, occupied_positions)
from bomber.framework.trader.runtime.live.channels.ctp_recovery import (
    CtpResumeLifecycle, validate_ctp_restored_positions)


def validate_restored_session(session, day):
    validate_ctp_restored_positions(session, day, session.recovery_binding)
    revision = restored_target_revision(session, session.recovery_binding, allow_empty=True)
    session.strategy.restore_target_revision(revision)
    return revision


class OptionControl:
    def check_frames(self, session):
        if session.runner.failure or getattr(session.client, "report_errors", ()):
            raise RuntimeError(session.runner.failure or "执行回报冲突")
        session.references.refresh()
        session.runner.poll_features()
        if not session.runner.session_ready():
            raise RuntimeError(f"期权会话未就绪: {session.readiness.check().as_dict()}")
        stale = session.runner.stale_instruments()
        if stale:
            session.runner.close_role_gate(f"必需行情失效: {stale}")
        session.runner.flush_frames()

    def verify(self, session):
        if session.runner.failure or getattr(session.client, "report_errors", ()):
            raise RuntimeError(session.runner.failure or "执行回报冲突")
        if session.strategy.frames == 0:
            raise RuntimeError("没有取得封闭分钟帧")
        if self.require_fills and session.strategy.fills_received == 0:
            raise RuntimeError("本次未取得归属成交，不能通过成交验收")

    def snapshot(self, session, context):
        return {"resume_ready": self.resume_ready,
            "bar_time_basis": "replay_test_mapped" if getattr(getattr(session, "bar_feed", None), "time_map", None) else "source_event",
            "valuation_day": self.valuation_day,
            "request_time_basis": "receive_wall_clock", "reference_time_basis": "observation_wall_clock",
            "trading_day": context.trading_day if context else None,
            "observed_td_trading_day": getattr(self, "_observed_td_day", None),
            "observed_md_trading_day": getattr(self, "_observed_md_day", None),
            "reference_manifest": {} if context is None or context.references is None else context.references.manifest,
            "orders_submitted": getattr(getattr(self, "driver", None), "submitted_orders", 0),
            "acceptance": "attributed_fills" if self.require_fills else "market_frames",
            "positions_retained_on_stop": True}


class OptionLocalLifecycle(OptionControl, LocalExecutionLifecycle):
    def __init__(self, trading_day, *, mode, valuation_day, resume=False, state_file=None,
                 ready_seconds=90, require_fills=False):
        super().__init__(trading_day, resume=resume, state_file=state_file, ready_seconds=ready_seconds)
        self.mode, self.valuation_day, self.require_fills = mode, valuation_day, require_fills
        self._observed_md_day = None

    @property
    def event_label(self):
        return "Recording目标" if self.mode == "recording" else "模型执行回报"

    def start(self, session, context):
        if self.resuming:
            restored = session.manager.restore()
            if restored is None:
                raise RuntimeError("原策略状态文件不存在")
            validate_finished_orders(session)
            owned = validate_owned_positions(session.runner.position_manager, session.recovery_binding,
                account_scope=session.position_scope)
            for instrument, quantity in owned.items():
                if abs(quantity) > getattr(session.execution.client, "position_client", session.execution.client).risk_manager.instrument_limits[instrument].max_abs_position:
                    raise RuntimeError("恢复策略仓位超过上限")
            revision = restored_target_revision(session, session.recovery_binding, allow_empty=True)
            session.strategy.restore_target_revision(revision)
        session.runner.start()
        wait_for_session_ready(session, self.ready_seconds)
        session.references.refresh()
        if session.manager is not None:
            session.manager.save()
        self.resume_ready = True
        session.runner.begin_bars()
        print(f"期权会话就绪: execution={self.mode} valuation_day={self.valuation_day}; 停机保留策略仓位", flush=True)

    def poll(self, session):
        self.check_frames(session)
        return True

    def events(self, session):
        if session is None:
            return ()
        return session.client.requests if self.mode == "recording" else session.execution.backend.reports

    def shutdown(self, session, context):
        if session is None:
            return {"cleanup_errors": []}
        session.strategy.stopping = True
        session.runner.accept_bars = False
        if session.upstream is not None:
            self._observed_md_day = session.upstream.latest_trading_day
        try:
            if session.manager is not None and (not self.resuming or self.resume_ready):
                session.manager.save()
        finally:
            session.runner.stop()
        return {"cleanup_errors": [], "final_gross": None, "final_active_orders": None,
            "resume_state_preserved": self.resuming and not self.resume_ready}


class OptionCtpLifecycle(OptionControl, CtpResumeLifecycle, CtpSessionLifecycle):
    def __init__(self, *args, valuation_day, state_file, require_fills=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.valuation_day, self.require_fills = valuation_day, require_fills
        self.bar_time_basis = "source_event"
        self.state_file = state_file

    def prepare(self, resources):
        resources.enter_context(strategy_state_lock(self.state_file))
        return super().prepare(resources)

    def validate_restored_session(self, session, day):
        return validate_restored_session(session, day)

    def start(self, session, context):
        if self.resuming:
            return super().start(session, context)
        self.session = session
        self._validate_position_scope(session)
        session.client.start()
        if self.account_reconciliation == "strict":
            assert_flat_account(self.driver, self.transport, position_scope=self.position_scope)
        else:
            assert_account_ready(self.driver, reconcile_positions=False)
        if self.driver.trading_day != context.trading_day or self.lost:
            raise RuntimeError("装配期间TD交易日或连接变化")
        session.runner.start()
        wait_for_session_ready(session, self.ready_seconds, lost=lambda: bool(self.lost))
        session.references.refresh()
        session.manager.save()
        session.client.arm_demo(session.client.DEMO_CONFIRMATION)
        self.resume_ready = True
        session.runner.begin_bars()
        self._next_check = time.monotonic() + self.reconcile_seconds

    def poll(self, session):
        if self.lost:
            raise RuntimeError(self.lost[-1])
        self.check_frames(session)
        if getattr(session.client, "portfolio_failure", None):
            raise RuntimeError(session.client.portfolio_failure)
        if self.driver.submitted_orders >= self.max_session_orders:
            return False  # 公共停机流程撤单、核验与保存，保留已成交仓位。
        if time.monotonic() >= self._next_check:
            session.client.refresh_account_state()
            if self.account_reconciliation != "off":
                gross = self._managed_gross(occupied_positions(self.transport))
                expected = self._expected_gross(session)
                self.position_differences = {key: {"observed": gross.get(key, (0, 0)),
                    "execution": expected.get(key, (0, 0))} for key in gross.keys() | expected.keys()
                    if gross.get(key, (0, 0)) != expected.get(key, (0, 0))}
                if self.position_differences and self.account_reconciliation == "strict":
                    raise RuntimeError(f"柜台与策略成交账本不一致: {self.position_differences}")
            self._next_check = time.monotonic() + self.reconcile_seconds
        return True

    @staticmethod
    def _expected_gross(session):
        result = {}
        for instrument in session.position_scope.instruments:
            from bomber.framework.market.basic.base import InstrumentId
            snap = session.ledger.snapshot(InstrumentId.from_str(instrument))
            if snap.long_total or snap.short_total:
                result[instrument] = (snap.long_total, snap.short_total)
        return result

    def shutdown(self, session, context):
        if session is not None:
            session.strategy.stopping = True
        return super().shutdown(session, context)
