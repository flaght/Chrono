import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf011(netFlowXL, netFlowS,  window=20, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = roller_corr(netFlowXL, netFlowS, window, 1, method)
    return alpha

    
