import pdb
from lumina.impulse.fixed import *


def gd002(open, high, low, close, volume, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'

    core1 = volume[((close - open).abs() <= (high - low).abs())
                   & ((close <= open))]

    core1 = roller_sum(core1.fillna(0), weriod, weriod, method)

    core1 /= roller_sum(volume, weriod, weriod, method)

    alpha = roller_mean(core1, window, window, method)

    return alpha
