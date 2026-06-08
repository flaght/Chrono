from lumina.impulse.fixed import *


##
#gamma2
def oi002(close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    turn_openint = openint / 1e6

    df_v1 = roller_std(close, window, 1, method)
    df_v2 = roller_std(turn_openint, window, 1, method)
    df_sign = df_v1.mask(df_v1 < 0, -1).mask(df_v1 >= 0, 1)
    df_v1 = df_v1.abs()
    df_sign_v2 = df_sign * df_v2

    df_corr = roller_corr(df_v1, df_sign_v2, weriod, 1, method)

    df_ystd = roller_std(df_v1, weriod, 1, method)
    df_xstd = roller_std(df_sign_v2, weriod, 1, method)

    alpha = df_corr * df_ystd / df_xstd

    alpha = roller_mean(alpha, window, 1, method)
    return alpha