from lumina.impulse.fixed import *


##
def oi001(close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    turn_openint = openint / 1e6

    df_sign = rets.mask(close < 0, -1).mask(rets >= 0, 1)
    df_v1 = rets.abs()
    df_sign_v2 = df_sign * turn_openint

    df_corr = roller_corr(df_v1, df_sign_v2, weriod, 1, method)
    df_ystd = roller_std(df_v1, weriod, 1, method)
    df_xstd = roller_std(df_sign_v2, weriod, 1, method)

    alpha = df_corr * df_ystd / df_xstd

    alpha = roller_mean(alpha, window, 1, method)
    return alpha
