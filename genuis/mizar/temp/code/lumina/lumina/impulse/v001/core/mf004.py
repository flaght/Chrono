import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf004(outflowXLRate,  ret, window=50, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    alpha = roller_corr(-1*outflowXLRate, ret, window, 1, method)
    return alpha

    
