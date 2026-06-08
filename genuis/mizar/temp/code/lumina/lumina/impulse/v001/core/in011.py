import pdb
import pandas as pd
from lumina.impulse.fixed import *


## obv
def in011(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    direction = np.sign(close.diff())
    obv = volume * direction
    obv = roller_sum(obv, weriod, 1, method)

    alpha = roller_mean(obv, window, 1, method)
    return alpha
