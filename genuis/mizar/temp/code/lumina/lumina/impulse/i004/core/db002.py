import pdb
from lumina.impulse.fixed import *


#factor_dongbei_core_8
def db002(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close)

    mean1 = roller_mean(rets, weriod, weriod, method)
    std1 = roller_std(rets, weriod, weriod, method)

    rets_flag = np.where(rets > (mean1 + std1), 1, np.nan)

    need = volume.mul(rets_flag)
    need = need.fillna(0)

    std2 = roller_std(need, weriod, 1, method)
    mean2 = roller_mean(need, weriod, 1, method)

    core1 = std2.div(mean2)

    alpha = roller_mean(core1, window, window, method)

    return alpha
