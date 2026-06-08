import pdb
import pandas as pd
from lumina.impulse.fixed import *


## EMA
def in003(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = roller_mean(close, weriod, 1, 'ewm')

    alpha = roller_mean(alpha, window, 1, method)
    return alpha
