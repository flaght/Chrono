import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf003(mainFlowRate,  window, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    
    mainflow_shift = mainFlowRate.shift(1) 
    alpha = roller_corr(mainFlowRate, mainflow_shift, window, 1, method)
    return alpha

    
