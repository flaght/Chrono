# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""高性能原生 Cython 实体工厂实现 (.pyx)。

适用于编译为 .pyd (Windows) 或 .so (Linux)。
利用 C-level 定点数直接初始化 TradeTick 与 QuoteTick，零 Python 字符串中转。
"""

from libc.stdint cimport uint8_t, uint64_t

from bomber.core.rust.model cimport AggressorSide
from bomber.model.data cimport QuoteTick, TradeTick
from bomber.model.identifiers cimport InstrumentId, TradeId
from bomber.model.objects cimport Price, Quantity


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
    # Bomber raw values always use FIXED_SCALAR (10^16 in high-precision mode),
    # independently of the display precision.  Let the native value objects do
    # that conversion; multiplying by 10^price_prec/size_prec produces values
    # close to zero when passed to ``from_raw_c``.
    cdef Price price_value = Price(price, price_prec)
    cdef Quantity size_value = Quantity(size, size_prec)
    cdef TradeId tid = TradeId(trade_id)

    return TradeTick.from_raw_c(
        instrument_id,
        price_value._mem.raw,
        price_value._mem.precision,
        size_value._mem.raw,
        size_value._mem.precision,
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

    cdef Price price_value
    cdef Quantity size_value
    cdef TradeId tid
    cdef AggressorSide agg = <AggressorSide>aggressor_side

    for i in range(n):
        price_value = Price(prices[i], price_prec)
        size_value = Quantity(sizes[i], size_prec)
        tid = TradeId(trade_ids[i])

        result[i] = TradeTick.from_raw_c(
            instrument_id,
            price_value._mem.raw,
            price_value._mem.precision,
            size_value._mem.raw,
            size_value._mem.precision,
            agg,
            tid,
            ts_events[i],
            ts_inits[i],
        )

    return result
