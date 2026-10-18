import pdb
from lumina.impulse.fixed import *


## donchian
def tv009(high, low, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    high_max = roller_max(high, weriod, weriod, 'rolling')
    low_min = roller_min(low, weriod, weriod, 'rolling')

    alpha = roller_mean((high_max + low_min) / 2, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha
