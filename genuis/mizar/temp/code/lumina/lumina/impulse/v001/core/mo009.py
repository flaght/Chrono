import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mo009(openint, long, short, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = (long - short) / openint
    alpha = roller_mean(alpha, window, 1, method)
    return alpha
