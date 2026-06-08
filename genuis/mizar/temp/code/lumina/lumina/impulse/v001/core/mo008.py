import pdb
import pandas as pd
from lumina.impulse.fixed import *

def mo008(openint, short, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    f1 = short / openint
    f2 = short.shift(weriod) / openint.shift(weriod)
    alpha = f1 / f2
    alpha = roller_mean(alpha, window, 1, method)
    return alpha