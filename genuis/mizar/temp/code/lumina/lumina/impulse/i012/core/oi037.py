from lumina.impulse.fixed import *


# obv
def oi037(close, openint, weriod, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    close_diff = close.diff(1)
    obv = np.sign(close_diff) * openint
    obv = roller_sum(obv, weriod, 1, method)

    alpha = roller_mean(obv, window, 1, method)

    return alpha
