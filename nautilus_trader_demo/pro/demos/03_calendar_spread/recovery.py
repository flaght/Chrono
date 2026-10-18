"""跨期固定会话的恢复与验收；复用公共成交账本及CTP核验。"""

import time

from bomber.framework.trader.runtime.live.channels.ctp import (
    CtpSessionLifecycle, assert_flat_account, assert_account_ready)
from bomber.framework.trader.runtime.live.channels.ctp_market import CtpMarketSessionLifecycle
from bomber.framework.trader.runtime.live.channels.ctp_recovery import (
    CtpResumeLifecycle, validate_ctp_restored_positions)
from bomber.framework.trader.runtime.live.local import strategy_state_lock
from bomber.framework.trader.runtime.live.readiness import wait_for_session_ready
from bomber.framework.trader.runtime.live.recovery import (
    validate_finished_orders, validate_owned_positions, restored_target_revision)
from .live_runtime import restore_direction


class CalendarControl:
    require_fills = False

    def verify(self, session):
        runner, strategy = session.runner, session.strategy
        failure = runner.failure or getattr(session.client, "portfolio_failure", None)
        if failure or getattr(session.client, "report_errors", ()):
            raise RuntimeError(failure or "执行回报冲突")
        if strategy.complete_frames < strategy.config.lookback + 1:
            raise RuntimeError(f"双腿同步预热不足: actual={strategy.complete_frames} "
                f"required={strategy.config.lookback + 1}")
        if self.resuming and strategy._force_target:
            raise RuntimeError("恢复后尚未发布重新预热的有效目标")
        if self.require_fills and not strategy.fills_received:
            raise RuntimeError("本次没有归属成交，不能通过成交验收")
        if self.require_fills:
            targets = strategy.last_targets
            binding = session.recovery_binding
            if targets is None or set(targets) != set(binding.targets):
                raise RuntimeError("成交验收缺少完整双腿最新目标")
            positions = runner.position_manager
            for key, instrument in binding.targets.items():
                actual = positions.position(strategy.strategy_id, key)
                working = positions.working_quantity(binding.client_id, instrument)
                unassigned = positions.unassigned_position(binding.client_id, instrument)
                if actual != targets[key] or working != 0 or unassigned != 0:
                    raise RuntimeError(f"双腿目标未完成: {key} target={targets[key]} "
                        f"position={actual} working={working} unassigned={unassigned}")

    def snapshot(self, session, context):
        fields = super().snapshot(session, context)
        fields.pop("fixed_instrument", None)
        fields.pop("factor_date", None)
        fields.pop("factor_date_basis", None)
        return {**fields, "acceptance": "attributed_fills" if self.require_fills else "synchronized_frames",
            "reference_manifest": context.references.manifest if context and context.references else {},
            "bar_time_basis": self.profile.bar_time_basis,
            "reference_time_basis": "current_reference_day_and_wall_clock",
            "positions_retained_on_stop": True,
            "fixed_instruments": {} if session is None else {
                r: str(i) for r, i in session.runner.instruments.items()},
            "pending_frames": [] if session is None else [{"bar_ns": stamp,
                "received": sorted(str(i) for i in frame),
                "missing": sorted(str(i) for i in session.runner.fixed_ids - set(frame))}
                for stamp, frame in sorted(session.runner.pending_frames.items())],
            "expired_frame_count": 0 if session is None else session.runner.expired_frame_count}


class CalendarLocalLifecycle(CalendarControl, CtpMarketSessionLifecycle):
    def __init__(self, *, mode, require_fills=False, **kwargs):
        super().__init__(**kwargs)
        self.mode, self.require_fills = mode, require_fills

    @property
    def event_label(self):
        return "Recording目标" if self.mode == "recording" else "模型执行回报"

    def start(self, session, context):
        session.runner.accept_bars = False
        if self.resuming:
            if session.manager.restore() is None:
                raise RuntimeError("跨期策略原状态文件不存在")
            validate_finished_orders(session)
            owned = validate_owned_positions(session.runner.position_manager, session.recovery_binding,
                account_scope=session.position_scope)
            if any(abs(v) > session.strategy.config.quantity for v in owned.values()):
                raise RuntimeError("恢复仓位超过策略手数上限")
            revision = restored_target_revision(session, session.recovery_binding, allow_empty=True)
            session.strategy.restore_target_revision(revision)
            restore_direction(session)
        session.runner.start()
        wait_for_session_ready(session, self.ready_seconds)
        if session.manager is not None:
            session.manager.save()
        self.resume_ready = True
        session.runner.begin_bars()
        print(f"跨期会话就绪: execution={self.mode}; 停机保留策略仓位", flush=True)

    def poll(self, session):
        if session.runner.failure or getattr(session.client, "report_errors", ()):
            raise RuntimeError(session.runner.failure or "执行回报冲突")
        session.references.refresh()
        session.runner._check()
        return True

    def events(self, session):
        if session is None:
            return ()
        return session.client.requests if self.mode == "recording" else session.execution.backend.reports

    def shutdown(self, session, context):
        if session:
            with session.runner._submit_lock:
                session.runner.accept_bars = False
        # 公共Local停机不依赖Recording的backend属性（仅最终计数需要避开）。
        if self.mode == "recording":
            self._shutdown_md_day = None if session is None else session.upstream.latest_trading_day
            self._md_shutdown = True
            if session:
                session.runner.stop()
            return {"cleanup_errors": [], "final_gross": None, "final_active_orders": None}
        return super().shutdown(session, context)

    def snapshot(self, session, context):
        fields = super().snapshot(session, context)
        if self.mode == "recording":
            fields.update(strategy_position_source="none", execution_origin="recording", model=None)
        return fields


class CalendarCtpLifecycle(CalendarControl, CtpResumeLifecycle, CtpSessionLifecycle):
    def __init__(self, *args, state_file, require_fills=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_file, self.require_fills = state_file, require_fills

    def prepare(self, resources):
        resources.enter_context(strategy_state_lock(self.state_file))
        return super().prepare(resources)

    def validate_restored_session(self, session, day):
        validate_ctp_restored_positions(session, day, session.recovery_binding)
        revision = restored_target_revision(session, session.recovery_binding, allow_empty=True)
        restore_direction(session)
        session.strategy.restore_target_revision(revision)
        return revision

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
        session.manager.save()
        session.client.arm_demo(session.client.DEMO_CONFIRMATION)
        self.resume_ready = True
        session.runner.begin_bars()
        self._next_check = time.monotonic() + self.reconcile_seconds

    def poll(self, session):
        if getattr(session.client, "portfolio_failure", None):
            session.runner.close_role_gate(session.client.portfolio_failure)
        return super().poll(session)

    @staticmethod
    def _expected_gross(session):
        result = {}
        for instrument in session.runner.fixed_ids:
            state = session.ledger.snapshot(instrument)
            if state.long_total or state.short_total:
                result[str(instrument)] = (state.long_total, state.short_total)
        return result

    def shutdown(self, session, context):
        if session:
            with session.runner._submit_lock:
                session.runner.accept_bars = False
        return super().shutdown(session, context)
