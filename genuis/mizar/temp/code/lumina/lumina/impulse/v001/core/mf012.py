import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf012(inflowMRate, fast_window=10,  slow_window=35, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha1 = roller_mean(inflowMRate, fast_window, 1, method)
    alpha2 = roller_mean(inflowMRate, slow_window, 1, method)
    alpha = alpha1 - alpha2
    return alpha

    
