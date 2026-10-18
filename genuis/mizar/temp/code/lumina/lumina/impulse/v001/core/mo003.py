import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mo003(long, short, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    f1 = long - short
    f2 = long.shift(weriod) - short.shift(weriod)
    alpha = f1 / f2 - 1

    alpha = roller_mean(alpha, window, 1, method)

    return alpha 
