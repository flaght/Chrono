"""执行诊断包装必须只委托一次，保留异常、原订单及原生生命周期。"""
from decimal import Decimal
from importlib import import_module
from types import SimpleNamespace as NS
import unittest

Audit = import_module("demos.08_option_vega.execution_audit").AuditedExecutionBackend


class Backend:
    driver = "original-driver"
    reports = ("original-report",)
    def __init__(self, fail=False): self.calls, self.fail = [], fail
    def submit_order(self, order):
        self.calls.append(order)
        if self.fail: raise RuntimeError("send failed")
        return "actual-order-id"


class ExecutionAuditTests(unittest.TestCase):
    def setup_audit(self, fail=False):
        backend = Backend(fail)
        audit = Audit(backend, NS(get=lambda i: NS(price=Decimal("29.4"), ts_event=90)), clock_ns=lambda: 100)
        quote = NS(instrument_id="MO", ts_event=1, ts_init=99,
            bid_price=NS(as_decimal=lambda: Decimal("29.2")), ask_price=NS(as_decimal=lambda: Decimal("29.6")),
            bid_size=NS(as_decimal=lambda: Decimal(25)), ask_size=NS(as_decimal=lambda: Decimal(16)))
        audit.observe_quote(quote)
        return backend, audit

    def test_delegates_same_order_once_and_records_actual_limit_and_quote(self):
        backend, audit = self.setup_audit()
        order = NS(instrument_id="MO", side="SELL", quantity=Decimal(1), price=Decimal("29.2"))
        self.assertEqual(audit.submit_order(order), "actual-order-id")
        self.assertEqual(backend.calls, [order])
        self.assertIs(backend.calls[0], order)
        self.assertEqual(order.price, Decimal("29.2"))
        row, = audit.snapshot
        self.assertTrue(row["crosses_observed_quote"])
        self.assertEqual(row["limit_price"], "29.2")
        self.assertEqual(row["reference_price"], "29.4")
        self.assertEqual(audit.driver, backend.driver)
        self.assertIs(audit.reports, backend.reports)

    def test_passive_buy_and_sell_prices_remain_unchanged(self):
        for side, price in (("SELL", "29.4"), ("BUY", "29.4")):
            backend, audit = self.setup_audit()
            order = NS(instrument_id="MO", side=side, quantity=Decimal(1), price=Decimal(price))
            audit.submit_order(order)
            self.assertFalse(audit.snapshot[0]["crosses_observed_quote"])
            self.assertEqual(backend.calls[0].price, Decimal(price))

    def test_missing_or_stale_quote_never_claims_marketable_price(self):
        backend, audit = self.setup_audit()
        audit._clock_ns = lambda: 31_000_000_100
        order = NS(instrument_id="MO", side="SELL", quantity=Decimal(1), price=Decimal("29.0"))
        audit.submit_order(order)
        self.assertIsNone(audit.snapshot[0]["crosses_observed_quote"])
        audit.submit_order(NS(instrument_id="other", side="SELL", quantity=1, price=Decimal("29.0")))
        self.assertIsNone(audit.snapshot[1]["crosses_observed_quote"])

    def test_send_exception_propagates_without_retry(self):
        backend, audit = self.setup_audit(fail=True)
        order = NS(instrument_id="MO", side="SELL", quantity=1, price=Decimal("29.2"))
        with self.assertRaisesRegex(RuntimeError, "send failed"): audit.submit_order(order)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(audit.snapshot[0]["status"], "SUBMIT_RAISED")


if __name__ == "__main__": unittest.main()
