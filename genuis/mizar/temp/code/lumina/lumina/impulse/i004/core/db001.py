import pdb
from lumina.impulse.fixed import *


#factor_dongbei_core_7
def db001(close, volume, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close)
    mean1 = roller_mean(rets, weriod, weriod, method)
    std1 = roller_std(rets, weriod, weriod, method)

    rets_flag = np.where(rets > (mean1 + std1), 1, np.nan)

    need = volume.mul(rets_flag)
    need = need.fillna(0)

    mean2 = roller_mean(need, weriod, 1, method)
    mean3 = roller_mean(volume, weriod, 1, method)

    core1 = mean2.div(mean3)

    alpha = roller_std(
        rets.mul(rets_flag).fillna(0), weriod, int(weriod), method).mul(core1)

    alpha = roller_mean(alpha, window, window, method)

    return alpha
