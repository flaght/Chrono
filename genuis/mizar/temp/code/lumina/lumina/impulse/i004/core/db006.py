import pdb
from lumina.impulse.fixed import *


# factor_dongfang_core_1
def db006(volume,vwap, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'
    mean_vwap = roller_mean(vwap, weriod, weriod, method)

    weight1 = roller_sum(vwap.mul(volume), weriod, weriod, method)

    weight1 /= roller_sum(volume, weriod, weriod, method)

    core1 = mean_vwap.div(weight1)
    alpha = np.log(core1)

    alpha = roller_mean(alpha, window, window, method)
    return alpha