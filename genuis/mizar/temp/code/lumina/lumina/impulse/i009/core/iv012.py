import pdb
import pandas as pd
from lumina.impulse.fixed import *

def iv012(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)
    weight = roller_sum(chg * volume, weriod, 1, method)
    vol1 = roller_sum(volume, weriod, 1, method)
    factors = - (weight / vol1)

    alpha = roller_mean(factors, window, 1, method)
    return alpha