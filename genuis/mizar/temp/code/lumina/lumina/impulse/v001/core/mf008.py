import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf008(mainFlow, ret,  window=20, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = roller_corr(mainFlow, ret, window, 1, method)
    return alpha

    
