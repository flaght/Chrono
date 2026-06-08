import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf005(mainFlowRate,  window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = mainFlowRate.diff()
    alpha = roller_mean(alpha, window, 1, method)
    return alpha

    
