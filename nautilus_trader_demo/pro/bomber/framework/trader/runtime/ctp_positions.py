"""显式接管单合约一手SimNow仓位；不恢复旧策略信号或旧订单。"""

from dataclasses import dataclass
from decimal import Decimal

from bomber.framework.market.basic.base import InstrumentId
from bomber.framework.trader.execution.ctp.ledger import CtpLedgerState, CtpPositionSnapshot


@dataclass(frozen=True)
class AdoptedCtpPosition:
    instrument: InstrumentId
    quantity: Decimal
    position_date: str
    position_cost: Decimal
    trading_day: str

    def ledger_state(self, multiplier):
        multiplier = Decimal(str(multiplier))
        if not multiplier.is_finite() or multiplier <= 0:
            raise ValueError("接管仓位乘数须为正且有限")
        amount = abs(self.quantity)
        basis = self.position_cost / (amount * multiplier)
        # SHFE/INE按PositionDate分别返回今昨持仓成本；不从净仓推断分桶。
        values = [Decimal(0)] * 4
        bases = [Decimal(0)] * 4
        index = (0 if self.quantity > 0 else 2) + (0 if self.position_date == "1" else 1)
        values[index], bases[index] = amount, basis
        snap = CtpPositionSnapshot(self.instrument, *values, *bases)
        return CtpLedgerState(self.trading_day, {self.instrument: snap})


def inspect_adopted_position(driver, transport, instrument, quantity):
    """两轮完整查询必须一致；只支持明确指定的一手单向投机仓。"""
    instrument = InstrumentId.from_str(str(instrument))
    quantity = Decimal(str(quantity))
    if not quantity.is_finite() or quantity not in {Decimal(-1), Decimal(1)} or str(instrument.venue) not in {"SHFE", "INE"}:
        raise ValueError("首版接管只支持SHFE／INE明确的一手单向仓")
    expected = (Decimal(1), Decimal(0)) if quantity > 0 else (Decimal(0), Decimal(1))
    snapshots = []
    for _ in range(2):
        net = {str(k): v for k, v in driver.reconcile().items() if v}
        account = driver.reconcile_account_state()
        if driver.reconcile_active_orders().orders:
            raise RuntimeError("接管前账户有活动订单，拒绝接管")
        if "CNY" not in account.balances or account.balances["CNY"].available <= 0:
            raise RuntimeError("接管前未取得可用CNY资金")
        gross = {str(k): v for k, v in transport.query_gross_positions().items() if any(v)}
        if gross != {str(instrument): expected} or net != {str(instrument): quantity}:
            raise RuntimeError(f"接管权威仓位与显式预期不符: gross={gross} net={net}")
        totals, selected = {}, []
        for row in transport.query_position_details():
            key = f"{row['InstrumentID']}.{row['ExchangeID']}"
            amount = row["Position"]
            long, short = totals.get(key, (Decimal(0), Decimal(0)))
            totals[key] = (long + amount, short) if row["PosiDirection"] == "2" else (long, short + amount)
            if amount:
                selected.append(row)
        if {k: v for k, v in totals.items() if any(v)} != gross or len(selected) != 1:
            raise RuntimeError("接管总仓和今昨明细不一致")
        row = selected[0]
        position_date = str(row.get("PositionDate", ""))
        if position_date not in {"1", "2"} or str(row.get("HedgeFlag", "")) != "1":
            raise RuntimeError("接管须明确PositionDate及投机HedgeFlag")
        try:
            today = Decimal(str(row["TodayPosition"]))
            cost = Decimal(str(row["PositionCost"]))
        except (KeyError, ArithmeticError, ValueError) as error:
            raise RuntimeError("接管缺少有效TodayPosition／PositionCost，请更新TD原生扩展") from error
        if today != (abs(quantity) if position_date == "1" else Decimal(0)):
            raise RuntimeError("接管PositionDate与TodayPosition不一致")
        if not cost.is_finite() or cost <= 0:
            raise RuntimeError("接管PositionCost须为正且有限，不猜测持仓成本")
        snapshots.append(AdoptedCtpPosition(instrument, quantity, position_date, cost, driver.trading_day))
    if snapshots[0] != snapshots[1]:
        raise RuntimeError("接管查询期间持仓／成本／交易日变化")
    if driver.reconcile_active_orders().orders:
        raise RuntimeError("接管核验结束时账户有活动订单")
    return snapshots[-1]
