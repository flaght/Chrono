import pdb
from lumina.impulse.fixed import *

#factor_dongbei_core_11
def db003(close, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close)

    core1 = roller_sum(rets**2, weriod, weriod, method)

    alpha = roller_mean(core1, window, window, method)

    return alpha