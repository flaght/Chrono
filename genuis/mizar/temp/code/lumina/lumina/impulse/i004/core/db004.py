import pdb
from lumina.impulse.fixed import *


#factor_dongbei_core_12 #factor_dongbei_core_13
def db004(close, quant, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close)

    core1 = -roller_quantile(rets, quant, weriod, 1, 'rolling')

    alpha = roller_mean(core1, window, window, method)
    return alpha