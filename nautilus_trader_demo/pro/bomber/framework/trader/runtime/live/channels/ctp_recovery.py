"""CTP同交易日恢复与权威持仓核验；策略通过钩子处理自己的信号状态。"""

from decimal import Decimal
import time

from ...ctp_account import position_buckets
from ..contracts import LiveRunContext
from ..recovery import validate_finished_orders, validate_owned_positions
from .ctp import account_lock, occupied_positions, readiness_details


def validate_ctp_restored_positions(session, day, binding):
    ledger = session.ledger.state()
    if ledger.trading_day != day or not set(ledger.positions) <= set(binding.targets.values()):
        raise RuntimeError("恢复仅支持相同交易日和合约账本")
    validate_owned_positions(session.runner.position_manager, binding,
                             account_scope=getattr(session, "position_scope", None))
    validate_finished_orders(session)


def validate_ctp_account(session, transport, binding):
    scope = getattr(session, "position_scope", None)
    positions = validate_owned_positions(session.runner.position_manager, binding, account_scope=scope)
    expected = {}
    for instrument, net in positions.items():
        ledger = session.ledger.snapshot(instrument)
        if net != ledger.net_position:
            raise RuntimeError("恢复时策略归属、柜台净仓与保存账本不一致")
        if ledger.long_total or ledger.short_total:
            expected[str(instrument)] = (ledger.long_total, ledger.short_total)
    mode = getattr(getattr(session.client, "account_reconciliation", None), "value", "strict")
    if mode != "strict":
        # 仍检查归属、成交账本、订单和资金；物理持仓不能改变策略事实。
        if session.driver.reconcile_active_orders().orders:
            raise RuntimeError("恢复要求柜台无活动订单")
        account = session.client.account_state
        if account is None or "CNY" not in account.balances or account.balances["CNY"].available <= 0:
            raise RuntimeError("恢复未取得可用CNY资金")
        return
    gross = occupied_positions(transport)
    if (scope.select(gross) if scope is not None else gross) != expected:
        raise RuntimeError("恢复时管理合约总仓与保存策略不一致" if scope is not None else
                           "恢复时全账户总仓与保存策略不一致")
    rows = transport.query_position_details()
    if scope is not None:
        rows = [row for row in rows if scope.contains(f"{row['InstrumentID']}.{row['ExchangeID']}")]
    buckets = position_buckets(rows, expected)
    for instrument in binding.targets.values():
        ledger = session.ledger.snapshot(instrument)
        bucket = buckets.get(str(instrument))
        if (bucket.quantities if bucket else (Decimal(0),) * 4) != (
                ledger.long_today, ledger.long_yesterday, ledger.short_today, ledger.short_yesterday):
            raise RuntimeError("恢复时权威今昨仓分桶与保存账本不一致")
    if session.driver.reconcile_active_orders().orders:
        raise RuntimeError("恢复要求柜台无活动订单")
    account = session.client.account_state
    if account is None or "CNY" not in account.balances or account.balances["CNY"].available <= 0:
        raise RuntimeError("恢复未取得可用CNY资金")


class CtpResumeLifecycle:
    """与CtpSessionLifecycle组合；不将账户未知持仓自动归给当前策略。"""
    def __init__(self, *args, resume=False, **kwargs):
        super().__init__(*args, **kwargs)
        if resume and (not self.orders or self.adopt_instrument is not None
                       or not self.expected_trading_day):
            raise ValueError("恢复须启用交易、声明TD日，且不能同时接管仓位")
        self.resuming = resume
        self.resume_ready = False
        self.restored_generation = None

    def prepare(self, resources):
        if not self.resuming:
            return super().prepare(resources)
        for namespace in dict.fromkeys((self.lock_namespace, *self.legacy_lock_namespaces)):
            resources.enter_context(account_lock(self.transport.account_id, namespace=namespace))
        self.driver.start(lambda report: None)
        day = self.driver.trading_day
        self._observed_td_day = day
        if day != self.expected_trading_day:
            raise RuntimeError("恢复时TD交易日不符")
        self.driver.reconcile_account_state()
        if self.driver.reconcile_active_orders().orders:
            raise RuntimeError("恢复要求柜台无活动订单")
        return LiveRunContext(day, resources)

    def validate_restored_session(self, session, day):
        raise NotImplementedError("入口须提供自身恢复状态校验")

    def recovery_binding(self, session):
        return session.recovery_binding

    def start(self, session, context):
        if not self.resuming:
            return super().start(session, context)
        self.session = session
        validate_scope = getattr(self, "_validate_position_scope", None)
        if callable(validate_scope):
            validate_scope(session)
        session.runner.accept_bars = False
        summary = session.manager.restore()
        if summary is None:
            raise RuntimeError("恢复文件不存在，不能作为空仓新会话启动")
        self.restored_generation = summary.generation
        revision = self.validate_restored_session(session, context.trading_day)
        session.client.start()  # 本地归属不能代替本次柜台对账。
        validate_ctp_account(session, self.transport, self.recovery_binding(session))
        if self.driver.trading_day != context.trading_day or self.lost:
            raise RuntimeError("恢复期间会话发生变化")
        session.runner.start()
        deadline = time.monotonic() + self.ready_seconds
        while not session.runner.session_ready():
            if self.lost or time.monotonic() >= deadline:
                raise RuntimeError(f"恢复后MD/TD未就绪，保持闭闸：{readiness_details(session)}")
            time.sleep(0.2)
        session.references.refresh()
        current = getattr(session.runner, "reference_is_current", None)
        reference_ok = current() if callable(current) else session.references.spec == session.runner.fixed_spec
        if not reference_ok or not session.runner.session_ready():
            raise RuntimeError("恢复准备后参考或行情失效，保持闭闸")
        session.manager.save()
        self.resume_ready = True
        session.client.arm_demo(session.client.DEMO_CONFIRMATION)
        session.runner.begin_bars()
        self._next_check = time.monotonic() + self.reconcile_seconds
        print(f"策略会话已恢复: generation={summary.generation} revision={revision} "
              f"TD={context.trading_day}；持仓保留，重新预热后发布新目标", flush=True)

    def shutdown(self, session, context):
        if self.resuming and not self.resume_ready:
            # 失败状态不得覆盖原文件；未知柜台订单也不自动撤销。
            errors = []
            if session:
                session.runner.accept_bars = False
                try:
                    session.runner.stop()
                except Exception as error:
                    errors.append(str(error))
            try:
                self.driver.stop()
            except Exception as error:
                errors.append(str(error))
            return {"final_gross": None, "final_active_orders": None,
                    "cleanup_errors": errors, "resume_state_preserved": True}
        return super().shutdown(session, context)

    def snapshot(self, session, context):
        return {**super().snapshot(session, context), "resume_ready": self.resume_ready,
                "restored_generation": self.restored_generation}
