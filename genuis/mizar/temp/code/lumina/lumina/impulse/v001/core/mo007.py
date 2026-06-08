import pdb
import pandas as pd
from lumina.impulse.fixed import *

def mo007(openint, long, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    f1 = long / openint
    f2 = long.shift(weriod) / openint.shift(weriod)
    alpha = f1 / f2
    alpha = roller_mean(alpha, window, 1, method)
    return alpha