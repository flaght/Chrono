# -*- coding: utf-8 -*-

from ._BizDayConventions cimport BizDayConventions as bdc

cpdef enum BizDayConventions:
    Following = bdc.Following
    ModifiedFollowing = bdc.ModifiedFollowing
    Preceding = bdc.Preceding
    ModifiedPreceding = bdc.ModifiedPreceding
    Unadjusted = bdc.Unadjusted
    HalfMonthModifiedFollowing = bdc.HalfMonthModifiedFollowing
    Nearest = bdc.Nearest