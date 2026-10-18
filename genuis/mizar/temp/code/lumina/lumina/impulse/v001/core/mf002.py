import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf002(netFlowSRate, fast_window, slow_window, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    
    fast_alpha = roller_mean(netFlowSRate, fast_window, 1, method)
    slow_alpha = roller_mean(netFlowSRate, slow_window, 1, method)
    alpha = fast_alpha - slow_alpha
    return alpha
    
