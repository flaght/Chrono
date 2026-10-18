import pdb
from lumina.impulse.fixed import *


#WCP
def tc022(close, high, low, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    half_length = round(0.5 * (weriod + 1))
    wcp = roller_mean(high, weriod, 1, method) + roller_mean(
        low, weriod, 1, method) + 2 * roller_mean(close, weriod, 1, method)
    wcp = roller_mean(wcp, weriod, 1, method) / 4

    alpha = roller_mean(wcp, window, 1, method)
    return alpha
