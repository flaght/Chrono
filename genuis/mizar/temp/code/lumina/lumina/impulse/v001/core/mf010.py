import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf010(mainFlow, ret,  window=20, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha1 = roller_cov(mainFlow, ret, window, 1, method) 
    alpha2 = roller_var(ret, window, 1, method)
    alpha = mainFlow - alpha1 / alpha2 * ret 
    return alpha

    
