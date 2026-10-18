import pdb
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

# alpha125
def ixy001(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    positive = rets.where(rets > 0, np.nan).where(rets < 0, 1)
    negative = rets.where(rets < 0, np.nan).where(rets > 0, 1)

    positive_price = roller_mean(positive * close, weriod, 1, method)
    negative_price = roller_mean(negative * close, weriod, 1, method)
    ft = roller_mean(positive_price, weriod, 1, method) / roller_mean(
        negative_price, weriod, 1, method)
    ft1 = roller_mean(ft, weriod, 1, method)
    alpha = -(ft1 - roller_median(ft, weriod, 1, 'rolling')) / (
        roller_max(ft, weriod, 1, 'rolling') -
        roller_min(ft, weriod, 1, 'rolling'))
    alpha = roller_mean(alpha, window, 1, method)
    return alpha
