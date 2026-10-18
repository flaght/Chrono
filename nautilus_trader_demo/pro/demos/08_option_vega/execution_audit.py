"""08原生执行诊断：记录实际OrderIntent与观测盘口，不改变定价或发送。"""
from decimal import Decimal
import json
import logging
from threading import RLock
import time

logger = logging.getLogger("OptionExecutionAudit")


class AuditedExecutionBackend:
    def __init__(self, backend, prices, *, max_quote_age_seconds=30, clock_ns=None):
        self._backend, self._prices = backend, prices
        self._clock_ns = clock_ns or time.time_ns
        self._max_quote_age_ns = int(max_quote_age_seconds * 1e9)
        self._quotes, self._records, self._lock = {}, [], RLock()

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def observe_quote(self, quote):
        row = {"bid": str(quote.bid_price.as_decimal()), "ask": str(quote.ask_price.as_decimal()),
            "bid_size": str(quote.bid_size.as_decimal()), "ask_size": str(quote.ask_size.as_decimal()),
            "source_ns": quote.ts_event, "received_ns": quote.ts_init}
        with self._lock:
            self._quotes[quote.instrument_id] = row

    @property
    def snapshot(self):
        with self._lock:
            return [{**row, "quote": None if row["quote"] is None else dict(row["quote"])} for row in self._records]

    def submit_order(self, order):
        now = self._clock_ns()
        reference = self._prices.get(order.instrument_id)
        with self._lock:
            quote = self._quotes.get(order.instrument_id)
            quote = None if quote is None else dict(quote)
            age = None if quote is None else now - quote["received_ns"]
            side = getattr(order.side, "value", order.side)
            crosses = None
            if quote is not None and order.price is not None and 0 <= age <= self._max_quote_age_ns:
                crosses = Decimal(str(order.price)) <= Decimal(quote["bid"]) if side == "SELL" else Decimal(str(order.price)) >= Decimal(quote["ask"])
            row = {"submitted_ns": now, "instrument": str(order.instrument_id), "side": side,
                "quantity": str(order.quantity), "limit_price": None if order.price is None else str(order.price),
                "reference_price": None if reference is None else str(reference.price),
                "reference_received_ns": None if reference is None else reference.ts_event,
                "quote": quote, "quote_age_ns": age, "crosses_observed_quote": crosses,
                "status": "SUBMIT_ATTEMPT", "client_order_id": None}
            self._records.append(row)
            del self._records[:-32]
        logger.info("CTP报单诊断: %s", json.dumps(row, ensure_ascii=False))
        try:
            result = self._backend.submit_order(order)
        except Exception as error:
            with self._lock:
                row.update(status="SUBMIT_RAISED", error_type=type(error).__name__)
            raise
        with self._lock:
            row.update(status="SUBMIT_RETURNED", client_order_id=result)
        return result
