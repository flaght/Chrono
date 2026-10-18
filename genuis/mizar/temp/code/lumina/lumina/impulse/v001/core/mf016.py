import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf016(mainFlowRate, mainBuyVol, mainSellVol, turnoverVol, window=10, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    main_net_vol_rate = (mainBuyVol - mainSellVol) / turnoverVol
    diver = mainFlowRate - main_net_vol_rate
    alpha = roller_mean(diver, window, 1, method)
    return alpha


    
