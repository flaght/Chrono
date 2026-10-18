import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf009(smainFlow, ret,  window=20, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = roller_corr(smainFlow, ret, window, 1, method)
    return alpha

    
