import pdb
import pandas as pd
from lumina.impulse.fixed import *


## 支撑位 压力位
def in001(high, low, close, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)

    pp = roller_mean(pp, window, 1, method)
    r1 = roller_mean(r1, window, 1, method)
    s1 = roller_mean(s1, window, 1, method)
    r2 = roller_mean(r2, window, 1, method)
    s2 = roller_mean(s2, window, 1, method)
    r3 = roller_mean(r3, window, 1, method)
    s3 = roller_mean(s3, window, 1, method)
    return pp, r1, s1, r2, s2, r3, s3
