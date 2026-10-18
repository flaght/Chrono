import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf018(mainFlowRate, ret, window=20, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    sum_ret = roller_sum(ret, window, 1, method)
    sum_flow = roller_sum(mainFlowRate, window, 1, method)
    alpha = sum_ret / (sum_flow + 1e-6)
    return alpha
    

    
