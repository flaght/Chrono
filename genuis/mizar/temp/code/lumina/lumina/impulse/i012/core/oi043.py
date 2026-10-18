from lumina.impulse.fixed import *


def oi043(close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)

    need = roller_mean(openint, weriod, 1, method) + roller_std(
        openint, weriod, 1, method)

    #factor = roller_mean(-chg.where((chg > 0) & (volume > need)), weriod, 1,
    #                     method)
    factor = pd_ewm_mean(-chg.where((chg > 0) & (openint > need)),
                         span=weriod,
                         min_periods=1)

    factor1 = roller_mean(factor, weriod, 1, method)
    factor2 = factor - factor1

    alpha1 = roller_mean(factor1, window, 1, method)
    alpha2 = roller_mean(factor2, window, 1, method)
    return alpha1, alpha2
