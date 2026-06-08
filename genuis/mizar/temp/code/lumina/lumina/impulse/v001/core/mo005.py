import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mo005(short, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = short.pct_change(weriod)

    alpha = roller_mean(alpha, window, 1, method)
    
    return alpha