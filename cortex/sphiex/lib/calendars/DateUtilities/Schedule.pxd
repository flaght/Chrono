# -*- coding: utf-8 -*-

from .Date cimport Date
from .Calendar cimport Calendar
from .Period cimport Period


cdef class Schedule(object):

    cdef public _effectiveDate
    cdef public _terminationDate
    cdef public Period _tenor
    cdef public Calendar _cal
    cdef public int _convention
    cdef public int _terminationConvention
    cdef public int _rule
    cdef public list _dates
    cdef public list _isRegular
    cdef public bint _endOfMonth
    cdef public Date _firstDate
    cdef public Date _nextToLastDate

    cpdef size_t size(self)
    cpdef bint isRegular(self, size_t i)
    cpdef Calendar calendar(self)
    cpdef Period tenor(self)
    cpdef bint endOfMonth(self)