from lumina.impulse.fixed import *



# cmf
def oi033(high, low, close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ad = 2 * close - high - low

    ad *= openint / (high - low)
    cmf = roller_mean(ad, weriod, 1, method) / roller_mean(openint, weriod, 1, method)

    alpha = roller_mean(cmf, window, 1, method)
    return alpha
