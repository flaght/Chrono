# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""高性能原生 Cython 实体工厂实现 (.pyx)。

适用于编译为 .pyd (Windows) 或 .so (Linux)。
利用 C-level 定点数直接初始化 TradeTick 与 QuoteTick，零 Python 字符串中转。
"""

from libc.math cimport round
from libc.stdint cimport uint8_t, uint64_t

from bomber.core.rust.model cimport AggressorSide, PriceRaw, QuantityRaw
from bomber.model.data cimport QuoteTick, TradeTick
from bomber.model.identifiers cimport InstrumentId, TradeId
from bomber.model.objects cimport Price, Quantity


cdef double[10] POW10 = [
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
    100000000.0,
    1000000000.0,
]


cpdef TradeTick fast_make_trade_tick(
    InstrumentId instrument_id,
    double price,
    double size,
    uint8_t price_prec,
    uint8_t size_prec,
    str trade_id,
    uint64_t ts_event,
    uint64_t ts_init,
    uint8_t aggressor_side=0,
):
    cdef double price_scalar = POW10[price_prec] if price_prec < 10 else 10.0 ** price_prec
    cdef double size_scalar = POW10[size_prec] if size_prec < 10 else 10.0 ** size_prec

    cdef PriceRaw price_raw = <PriceRaw>round(price * price_scalar)
    cdef QuantityRaw size_raw = <QuantityRaw>round(size * size_scalar)
    cdef TradeId tid = TradeId(trade_id)

    return TradeTick.from_raw_c(
        instrument_id,
        price_raw,
        price_prec,
        size_raw,
        size_prec,
        <AggressorSide>aggressor_side,
        tid,
        ts_event,
        ts_init,
    )


cpdef QuoteTick fast_make_quote_tick(
    InstrumentId instrument_id,
    double bid_price,
    double ask_price,
    double bid_size,
    double ask_size,
    uint8_t price_prec,
    uint8_t size_prec,
    uint64_t ts_event,
    uint64_t ts_init,
):
    cdef Price bp = Price(bid_price, price_prec)
    cdef Price ap = Price(ask_price, price_prec)
    cdef Quantity bs = Quantity(bid_size, size_prec)
    cdef Quantity ass = Quantity(ask_size, size_prec)

    return QuoteTick(
        instrument_id,
        bp,
        ap,
        bs,
        ass,
        ts_event,
        ts_init,
    )


cpdef list fast_make_trade_ticks_from_arrays(
    InstrumentId instrument_id,
    uint8_t price_prec,
    uint8_t size_prec,
    const double[:] prices,
    const double[:] sizes,
    const uint64_t[:] ts_events,
    const uint64_t[:] ts_inits,
    list trade_ids,
    uint8_t aggressor_side=0,
):
    cdef Py_ssize_t n = prices.shape[0]
    cdef list result = [None] * n
    cdef Py_ssize_t i

    cdef double price_scalar = POW10[price_prec] if price_prec < 10 else 10.0 ** price_prec
    cdef double size_scalar = POW10[size_prec] if size_prec < 10 else 10.0 ** size_prec

    cdef PriceRaw p_raw
    cdef QuantityRaw s_raw
    cdef TradeId tid
    cdef AggressorSide agg = <AggressorSide>aggressor_side

    for i in range(n):
        p_raw = <PriceRaw>round(prices[i] * price_scalar)
        s_raw = <QuantityRaw>round(sizes[i] * size_scalar)
        tid = TradeId(trade_ids[i])

        result[i] = TradeTick.from_raw_c(
            instrument_id,
            p_raw,
            price_prec,
            s_raw,
            size_prec,
            agg,
            tid,
            ts_events[i],
            ts_inits[i],
        )

    return result
