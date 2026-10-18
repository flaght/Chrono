from lumina.impulse.fixed import *


def oi028(open, high, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    core1 = roller_corr(openint, high, weriod, weriod, method)
    value1 = np.arctan(openint).mul(core1)
    value1 = np.where(value1 <= 0, np.nan, value1)
    core1 = np.minimum(np.log(value1), open.diff(6))

    alpha = roller_mean(core1, window, window, method)

    return alpha
