import pdb
from lumina.impulse.fixed import *


# factor_dongbei_core_14  factor_dongbei_core_15
def db005(close, quant, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close)

    var1 = -roller_quantile(rets, quant, weriod, 1, 'rolling')

    core1 =  roller_mean(-(rets.mask(rets >= -var1)).fillna(0), weriod, weriod, method)

    alpha = roller_mean(core1, window, window, method)

    return alpha