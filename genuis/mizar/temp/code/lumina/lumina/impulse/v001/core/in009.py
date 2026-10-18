import pdb
import pandas as pd
from lumina.impulse.fixed import *


## vwap
def in009(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = roller_sum((close * volume), weriod, 1, method) / roller_sum(
        volume, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha
