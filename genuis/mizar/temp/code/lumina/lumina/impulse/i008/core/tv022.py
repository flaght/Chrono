import pdb
from lumina.impulse.fixed import *

## ichimoku


def tv022(high, low, window, tenkan, kijun, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    tenkan_lowest_low = roller_min(low, tenkan, 1, 'rolling')
    tenkan_highest_high = roller_max(high, tenkan, 1, 'rolling')
    tenkan_midprice = roller_mean(
        (tenkan_lowest_low + tenkan_highest_high) / 2, tenkan, 1, method)

    kijun_lowest_low = roller_min(low, kijun, 1, 'rolling')
    kijun_highest_high = roller_max(high, kijun, 1, 'rolling')
    kijun_midprice = roller_mean((kijun_lowest_low + kijun_highest_high) / 2,
                                 kijun, 1, method)

    span_a = 0.5 * (tenkan_midprice + kijun_midprice)

    alpha = roller_mean(span_a, window, 1, method)
    return alpha
