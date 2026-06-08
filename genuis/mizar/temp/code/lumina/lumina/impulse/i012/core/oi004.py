from lumina.impulse.fixed import *


#illiq3
def oi004(close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    turn_openint = openint /1e6

    rets_std = roller_std(rets, weriod, 1, method)
    volume_std = roller_std(turn_openint, weriod, 1, method)

    rets_mean = roller_mean(rets, weriod, 1, method)
    volume_mean = roller_mean(turn_openint, weriod, 1, method)

    f1 = safe_div(rets_mean, volume_mean)

    df_corr = roller_corr(rets_std, volume_std, weriod, 1, method)
    df_b = df_corr * rets_std / volume_std
    df_cv = volume_std / volume_mean

    alpha = f1 + df_b * df_cv**2

    alpha = roller_mean(alpha, window, 1, method)

    return alpha
