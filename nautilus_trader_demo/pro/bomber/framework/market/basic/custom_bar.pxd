# cython: language_level=3
"""复合 K 线 (CustomBar) Cython 声明文件 (.pxd)。

遵循 code/doc/MARKET_DATA_CYTHON_EVALUATION.md 规范：
作为高频流转实体（Hot Data Payload），继承原生 C 扩展 Data 类，实现紧凑 C 级内存存储与零 __dict__ 开销。
"""

from libc.stdint cimport uint64_t

# 尝试从原生 Cython 包导入基类
from bomber.model.data cimport Bar, Data
from bomber.model.identifiers cimport InstrumentId


cdef class CustomBar(Data):
    cdef readonly Bar bar
    cdef readonly dict factors
    cdef readonly uint64_t ts_event
    cdef readonly uint64_t ts_init

    cpdef double get_factor(self, str name, double default_value=*)
    cpdef bint has_factor(self, str name)
