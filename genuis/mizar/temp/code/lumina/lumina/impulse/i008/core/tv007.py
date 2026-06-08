import pdb
from lumina.impulse.fixed import *


# obv
def tv007(close, volume, weriod, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    close_diff = close.diff(1)
    obv = np.sign(close_diff) * volume
    obv = roller_sum(obv, weriod, 1, method)

    alpha = roller_mean(obv, window, 1, method)

    return alpha
