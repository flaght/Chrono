import pdb
from lumina.impulse.fixed import *


def rv012(close, threshold, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)

    core1 = roller_quantile(chg, threshold, weriod, 1, 'rolling')
    rsv = -chg.mask(chg < core1, np.nan)
    alpha = roller_mean(rsv, weriod, 1, method)


    factor1 = roller_mean(alpha, weriod, 1, method)
    factor2 = alpha - factor1

    alpha1 = roller_mean(factor1, window, 1, method)
    alpha2 = roller_mean(factor2, window, 1, method)

    return alpha1, alpha2