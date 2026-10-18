import pdb
import pandas as pd
from lumina.impulse.fixed import *


### 非流动性
def dv012(high, low, close, open, value, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    short_value = (2 * (high - low) - abs(close - open)) / (value / 1e6)

    alpha = roller_skew(short_value, weriod, 1, 'rolling')

    alpha = roller_mean(alpha, window, 1, method)

    return alpha
