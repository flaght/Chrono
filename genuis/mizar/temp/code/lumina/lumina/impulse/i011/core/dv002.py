import pdb
import pandas as pd
from lumina.impulse.fixed import *


## T分布主动占比
def dv002(close, value, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    std1 = roller_std(close, weriod, 1, method)
    alpha = roller_sum(value * (rets / std1), weriod, 1, method) / roller_sum(
        value, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha
