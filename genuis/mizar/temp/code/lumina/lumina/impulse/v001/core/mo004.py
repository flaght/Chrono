import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mo004(long, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    f1 = long.pct_change(weriod)

    alpha = roller_mean(f1, window, 1, method)
    
    return alpha