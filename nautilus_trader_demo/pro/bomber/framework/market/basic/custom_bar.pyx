# cython: language_level=3
"""复合 K 线 (CustomBar) Cython 实现文件 (.pyx)。

适用于编译为 .pyd (Windows) 或 .so (Linux)。
可被 NautilusTrader 的 MessageBus、DataEngine 和 Strategy 直接接收消费。
"""

from libc.stdint cimport uint64_t

from bomber.model.data cimport Bar, Data
from bomber.model.identifiers cimport InstrumentId


cdef class CustomBar(Data):
    """
    符合 NautilusTrader 内核规范的复合 K 线 C 扩展实体。
    零 __dict__ 字典开销，作为原生 Data 流经核心事件总线。
    """

    def __init__(
        self,
        Bar bar not None,
        dict factors not None,
        uint64_t ts_event,
        uint64_t ts_init,
    ) -> None:
        self.bar = bar
        self.factors = factors
        self.ts_event = ts_event
        self.ts_init = ts_init

    @property
    def instrument_id(self) -> InstrumentId:
        if hasattr(self.bar, "instrument_id"):
            return self.bar.instrument_id
        return self.bar.bar_type.instrument_id

    cpdef double get_factor(self, str name, double default_value=0.0):
        """C 级快速获取伴生因子数值。"""
        return self.factors.get(name, default_value)

    cpdef bint has_factor(self, str name):
        """C 级快速判断因子是否存在。"""
        return name in self.factors

    def __repr__(self) -> str:
        return (
            f"CustomBar(instrument_id={self.instrument_id}, "
            f"close={self.bar.close}, factors_count={len(self.factors)}, "
            f"ts_event={self.ts_event})"
        )
