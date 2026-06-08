from lumina.impulse.fixed import *


def oi022(close, openint, quant, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close)

    sum_int = roller_sum(openint, weriod, weriod, method)
    var1 = rets.mul(openint).div(sum_int)

    wvar1 = roller_quantile(var1, quant, weriod, 1, 'rolling')

    core1 = roller_mean(
        var1.mask(var1 < wvar1).fillna(0), weriod, weriod, method)

    alpha = roller_mean(core1, window, window, method)

    return alpha
