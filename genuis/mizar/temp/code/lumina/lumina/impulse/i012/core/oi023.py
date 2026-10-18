from lumina.impulse.fixed import *


def oi023(open, high, low, close, openint, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'

    core1 = openint[((close - open).abs() <= (high - low).abs())
                    & ((close <= open))]

    core1 = roller_sum(core1.fillna(0), weriod, weriod, method)

    core1 /= roller_sum(openint, weriod, weriod, method)

    alpha = roller_mean(core1, window, window, method)

    return alpha
