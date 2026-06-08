import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf017(smainFlowRate, ret, window=5, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    panic_sell = (0 * smainFlowRate) + np.where(ret < -0.02, -smainFlowRate, 0)
    alpha = roller_sum(panic_sell, window, 1, method)
    return alpha
    

    
