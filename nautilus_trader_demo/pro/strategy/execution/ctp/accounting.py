"""把统一成交回报写入CTP今昨仓账本。"""

from __future__ import annotations

from decimal import Decimal
from typing import Mapping

from market.basic.base import InstrumentId
from strategy.execution.contracts import ExecutionReport, ExecutionReportType
from strategy.execution.ctp.ledger import (
    CtpFillResult,
    CtpPositionLedger,
    CtpSettlementResult,
)
from strategy.execution.ctp.profile import CtpCommissionRule


class CtpExecutionAccounting:
    def __init__(
        self,
        ledger: CtpPositionLedger,
        multipliers: Mapping[InstrumentId, Decimal | int | float | str],
        commission_rules: Mapping[InstrumentId, CtpCommissionRule] | None = None,
    ) -> None:
        self.ledger = ledger
        self.multipliers = {
            instrument_id: Decimal(str(value))
            for instrument_id, value in multipliers.items()
        }
        self.commission_rules = dict(commission_rules or {})
        self.results: list[CtpFillResult] = []
        self.commissions: list[Decimal] = []
        self.settlements: list[CtpSettlementResult] = []

    @property
    def total_commission(self) -> Decimal:
        return sum(self.commissions, Decimal(0))

    @property
    def total_realized_pnl(self) -> Decimal:
        return sum((result.realized_pnl for result in self.results), Decimal(0))

    @property
    def total_variation_margin(self) -> Decimal:
        return sum(
            (result.variation_margin for result in self.settlements),
            Decimal(0),
        )

    @property
    def net_cash_change(self) -> Decimal:
        return (
            self.total_realized_pnl
            + self.total_variation_margin
            - self.total_commission
        )

    def settle(
        self,
        trading_day: str,
        settlement_prices: Mapping[InstrumentId, Decimal | int | float | str],
    ) -> CtpSettlementResult:
        result = self.ledger.settle(
            trading_day,
            settlement_prices,
            self.multipliers,
        )
        self.settlements.append(result)
        return result

    def on_report(self, report: ExecutionReport) -> None:
        if report.report_type not in {
            ExecutionReportType.PARTIALLY_FILLED,
            ExecutionReportType.FILLED,
        }:
            return
        if (
            report.order_side is None
            or report.position_effect is None
            or report.fill_price is None
        ):
            raise ValueError("CTP成交回报缺少side、position_effect或fill_price")
        multiplier = self.multipliers.get(report.instrument_id)
        if multiplier is None:
            raise ValueError(f"缺少CTP合约乘数: {report.instrument_id}")
        self.results.append(
            self.ledger.apply_fill(
                report.instrument_id,
                report.order_side,
                report.position_effect,
                report.filled_quantity,
                report.fill_price,
                multiplier,
            ),
        )
        rule = self.commission_rules.get(report.instrument_id)
        self.commissions.append(
            Decimal(0)
            if rule is None
            else rule.calculate(report.position_effect, report.filled_quantity),
        )
