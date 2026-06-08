import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf014(inflowL, inflowXL, buyOrdL, buyOrdXL, window=20, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    
    avg_L_ord = (inflowL + inflowXL) / (buyOrdL + buyOrdXL)
    alpha = -1 * (avg_L_ord - roller_mean(avg_L_ord, window, 1, method)) / roller_std(avg_L_ord, window, 1, method)
    return alpha

    
