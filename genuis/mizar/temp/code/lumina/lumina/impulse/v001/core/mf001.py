import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf001(mainFlow, smainFlow, fast_window, slow_window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = mainFlow - smainFlow 
    
    fast_alpha = roller_mean(alpha, fast_window, 1, method)
    slow_alpha = roller_mean(alpha, slow_window, 1, method)
    alpha = fast_alpha - slow_alpha
    return alpha
    
