import pdb
from lumina.impulse.fixed import *


# err
def iv009(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)

    med = roller_median(chg, weriod, 1, 'rolling')
    df_s = chg - med
    df_smax = roller_max(df_s, weriod, 1, 'rolling')

    df = chg.mask(df_s != df_smax, 0)
    df = roller_max(df, weriod, 1, 'rolling')

    df_pre = chg.shift(1).mask(df_s != df_smax, 0)
    df_pre = roller_max(df_pre, weriod, 1, 'rolling')

    factors = df - df_pre

    alpha = roller_mean(factors, window, 1, method)

    return alpha
