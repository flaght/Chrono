import pdb
import pandas as pd
from lumina.impulse.fixed import *


## ATR
def in008(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    high_low = high - low
    high_close_prev = (high - close.shift(weriod)).abs()
    low_close_prev = (low - close.shift(weriod)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev],
                           axis=1).max(axis=1).to_frame()
    true_range.columns = high_low.columns
    atr = roller_mean(true_range, window, 1, method)
    return atr
