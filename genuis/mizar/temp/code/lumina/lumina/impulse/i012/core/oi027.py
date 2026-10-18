from lumina.impulse.fixed import *


def oi027(high, openint, value, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'

    oni = openint / 1e6
    reg_beta = roller_cov(value, high, weriod, weriod,
                          method).div(roller_std(value, weriod, weriod,
                                                 method))

    core1 = roller_cov(oni, value, weriod, weriod, method)

    core1 /= roller_mean(reg_beta, weriod, weriod, method)

    alpha = roller_mean(core1, window, window, method)

    return alpha
