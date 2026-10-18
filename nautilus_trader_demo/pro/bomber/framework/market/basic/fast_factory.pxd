# cython: language_level=3
"""高性能原生 Cython 实体工厂声明 (.pxd)。

直接调用 NautilusTrader / Bomber 的 C 级定点数构造函数 (from_raw_c)，
彻底绕过 Python 浮点格式化与字符串解析，实现 C 内存直通构造。
"""

from libc.stdint cimport uint8_t, uint64_t
from bomber.core.rust.model cimport AggressorSide, PriceRaw, QuantityRaw
from bomber.model.data cimport QuoteTick, TradeTick
from bomber.model.identifiers cimport InstrumentId, TradeId


cpdef TradeTick fast_make_trade_tick(
    InstrumentId instrument_id,
    double price,
    double size,
    uint8_t price_prec,
    uint8_t size_prec,
    str trade_id,
    uint64_t ts_event,
    uint64_t ts_init,
    uint8_t aggressor_side=*,
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
    uint8_t aggressor_side=*,
)
