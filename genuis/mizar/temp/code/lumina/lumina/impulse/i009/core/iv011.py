import pdb
import pandas as pd
from lumina.impulse.fixed import *


#RHO
def iv011(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)
    df_x = chg.shift(1)
    df_y = chg.copy()

    ## 斜率
    df_corr = roller_corr(df_x, df_y, weriod, 1, method)
    df_ystd = roller_std(df_y, weriod, 1, method)
    df_xstd = roller_std(df_x, weriod, 1, method)
    df_slope = df_corr * df_ystd / df_xstd

    ##残差
    df_resid = df_y - df_slope * df_x
    factor = roller_corr(df_resid, df_resid.shift(1), weriod, 1, method)
    alpha = roller_mean(factor, window, 1, method)

    return alpha
