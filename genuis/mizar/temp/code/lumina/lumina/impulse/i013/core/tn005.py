import pdb
import pandas as pd
from lumina.impulse.fixed import *


##投资者情绪指标 ## 加benchmark
def tn005(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    vol_chg = safe_log(volume)
    close_chg = safe_log(close)

    p1 = roller_cov(vol_chg, close_chg, weriod, 1, method)
    p2 = roller_var(close_chg, weriod, 1, method)

    alpha = p1 / p2

    alpha = roller_mean(alpha, window, 1, method)
    return alpha
