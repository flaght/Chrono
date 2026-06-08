import pdb
import pandas as pd
from lumina.impulse.fixed import *


## bollinger bands
def in006(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    middle = roller_mean(close, weriod, 1, method)
    std1 = roller_std(close, weriod, 1, method)
    upper = middle + (std1 * 2)
    lower = middle - (std1 * 2)

    middle = roller_mean(middle, window, 1, method)
    upper = roller_mean(upper, window, 1, method)
    lower = roller_mean(lower, window, 1, method)

    return middle, upper, lower
