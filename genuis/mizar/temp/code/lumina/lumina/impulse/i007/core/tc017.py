import pdb
from lumina.impulse.fixed import *


## rsi
def tc017(close, window, weriod, scalar=None, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    scalar = float(scalar) if scalar else 1000

    negative = close.diff()
    positive = negative.copy()

    negative[negative > 0] = 0
    positive[positive < 0] = 0

    positive_avg = roller_mean(negative, weriod, 1, method)
    negative_avg = roller_mean(positive, weriod, 1, method)

    rsi = scalar * positive_avg / (positive_avg + negative_avg.abs())

    alpha = roller_mean(rsi, window, 1, method)

    return alpha
