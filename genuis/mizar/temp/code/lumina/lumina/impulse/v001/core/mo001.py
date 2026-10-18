import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mo001(long, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = long.diff(weriod)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha
