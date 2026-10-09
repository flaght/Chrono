"""04同交易日、已结束订单的组合恢复；不接管未知仓位。"""

import time
from decimal import Decimal

from bomber.framework.trader.runtime.ctp import account_lock, occupied_positions
from bomber.framework.trader.runtime.managed import LiveRunContext


def validate_restored_session(session, day):
    runner, client = session.runner, session.client
    keys = {str(i) for i in runner.fixed_ids}
    targets = runner.target_store.all()
    if set(targets) != {session.strategy.strategy_id}:
        raise RuntimeError("恢复文件不是当前单策略组合")
    target = targets[session.strategy.strategy_id]
    if set(target.targets) != keys:
        raise RuntimeError("恢复文件合约与当前主力组合不一致")
    ledger = session.ledger.state()
    if ledger.trading_day != day or not set(ledger.positions) <= runner.fixed_ids:
        raise RuntimeError("恢复仅支持相同交易日和合约账本")
    snapshot = runner.position_manager.snapshot()
    if any(sid != session.strategy.strategy_id or key not in keys
           for sid, key in snapshot.strategy_positions):
        raise RuntimeError("恢复文件存在其他策略或逻辑仓位")
    if any(snapshot.working_quantities.values()) or any(
            not order.status.is_terminal for order in client.order_state_machine.states()):
        raise RuntimeError("恢复要求原会话活动订单已结束")
    # 目标版本延续；动量窗口重新预热，不能重置到revision=1。
    with session.strategy._revision_lock:
        session.strategy._revision = target.revision
    return target.revision


def validate_account(session, transport):
    positions = session.runner.position_manager
    expected = {}
    for instrument in session.runner.fixed_ids:
        owned = positions.position(session.strategy.strategy_id, str(instrument))
        net = positions.account_position(session.client.client_id, instrument)
        ledger = session.ledger.snapshot(instrument)
        if (owned != net or net != ledger.net_position or
                positions.unassigned_position(session.client.client_id, instrument)):
            raise RuntimeError("恢复时策略归属、柜台净仓与保存账本不一致")
        if ledger.long_total or ledger.short_total:
            expected[str(instrument)] = (ledger.long_total, ledger.short_total)
    if occupied_positions(transport) != expected:
        raise RuntimeError("恢复时全账户总仓与保存组合不一致")
    buckets = {}
    for row in transport.query_position_details():
        qty = Decimal(str(row["Position"]))
        if not qty:
            continue
        key = row["InstrumentID"] + "." + row["ExchangeID"]
        direction, position_date = str(row.get("PosiDirection")), str(row.get("PositionDate"))
        if (key not in expected or str(row.get("HedgeFlag")) != "1" or
                direction not in {"2", "3"} or position_date not in {"1", "2"} or
                not qty.is_finite() or qty < 0):
            raise RuntimeError("恢复时权威今昨仓明细无效")
        index = (0 if direction == "2" else 2) + (0 if position_date == "1" else 1)
        values = buckets.setdefault(key, [Decimal(0)] * 4)
        values[index] += qty
    for instrument in session.runner.fixed_ids:
        ledger = session.ledger.snapshot(instrument)
        if tuple(buckets.get(str(instrument), [Decimal(0)] * 4)) != (
                ledger.long_today, ledger.long_yesterday, ledger.short_today, ledger.short_yesterday):
            raise RuntimeError("恢复时权威今昨仓分桶与保存账本不一致")
    if session.driver.reconcile_active_orders().orders:
        raise RuntimeError("恢复要求柜台无活动订单")
    account = session.client.account_state
    if account is None or "CNY" not in account.balances or account.balances["CNY"].available <= 0:
        raise RuntimeError("恢复未取得可用CNY资金")


class ResumeLifecycle:
    def __init__(self, *args, resume=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.resuming = resume
        self.resume_ready = False

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

    def start(self, session, context):
        if not self.resuming:
            return super().start(session, context)
        self.session = session
        summary = session.manager.restore()
        self.restored_generation = summary.generation
        revision = validate_restored_session(session, context.trading_day)
        session.client.start()  # 原生检查账户身份、订单关联、成交去重，再权威对账。
        validate_account(session, self.transport)
        if self.driver.trading_day != context.trading_day or self.lost:
            raise RuntimeError("恢复期间会话发生变化")
        session.runner.start()
        deadline = time.monotonic() + self.ready_seconds
        while not session.runner.session_ready():
            if self.lost or time.monotonic() >= deadline:
                raise RuntimeError("恢复后MD/TD未就绪，保持闭闸")
            time.sleep(0.2)
        session.references.refresh()
        if session.references.spec != session.runner.fixed_spec or not session.runner.session_ready():
            raise RuntimeError("恢复准备后参考或行情失效，保持闭闸")
        self.resume_ready = True
        session.manager.save()
        session.client.arm_demo(session.client.DEMO_CONFIRMATION)
        session.runner.begin_bars()
        self._next_check = time.monotonic() + self.reconcile_seconds
        print(f"组合已恢复: generation={summary.generation} revision={revision} "
              f"TD={context.trading_day}；持仓保留，实时重新预热后发布新目标", flush=True)
