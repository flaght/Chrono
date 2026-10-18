import pdb
from lumina.impulse.fixed import *

#emo
def tv005(high, low, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    high_low_range = high - low
    distance =  0.5 * (high + low)
    distance -= (0.5 * (high.shift(1) + low.shift(1)))
    box_ratio = volume / 10e6
    box_ratio = box_ratio / high_low_range
    emo = distance / box_ratio
    emo = roller_mean(emo, weriod, 1, method)

    alpha = roller_mean(emo, window, 1, method)
    return alpha
