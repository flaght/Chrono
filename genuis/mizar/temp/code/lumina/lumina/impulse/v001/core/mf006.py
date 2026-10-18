import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf006(inflowXL, outflowXL,  window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = inflowXL / (-outflowXL+1e-6)
    alpha = roller_mean(alpha, window, 1, method)
    return alpha

    
