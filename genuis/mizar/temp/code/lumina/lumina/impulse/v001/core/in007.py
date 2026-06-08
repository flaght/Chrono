import pdb
import pandas as pd
from lumina.impulse.fixed import *


## KDJ
def in007(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    low_min = pd_rolling_min(close, window=weriod, min_periods=1)
    high_max = pd_rolling_max(close, window=weriod, min_periods=1)

    rsv = 100 * ((close - low_min) / (high_max - low_min))
    k = roller_mean(rsv, weriod, 1, 'ewm')
    d = roller_mean(k, weriod, 1, 'ewm')
    j = 3 * k - 2 * d

    k = roller_mean(k, window, 1, method)
    d = roller_mean(d, window, 1, method)
    j = roller_mean(j, window, 1, method)
    return k, d, j
