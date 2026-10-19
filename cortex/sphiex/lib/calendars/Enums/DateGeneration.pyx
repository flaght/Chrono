# -*- coding: utf-8 -*-

from ._DateGeneration cimport DateGeneration as dg


cpdef enum DateGeneration:
    Zero = dg.Zero
    Backward = dg.Backward
    Forward = dg.Forward