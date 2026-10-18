from lumina.impulse.fixed import *


def oi025(openint, value, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    vwap = value.div(openint)
    mean_value = roller_mean(value, weriod, weriod, method)

    core1_vwap = roller_mean(vwap, weriod, weriod, method)

    core1_vwap = roller_max(core1_vwap, weriod, int(weriod / 2), 'rolling')

    core1 = roller_cov(core1_vwap, mean_value, weriod, weriod, method)

    alpha = roller_mean(core1, window, window, method)

    return alpha
