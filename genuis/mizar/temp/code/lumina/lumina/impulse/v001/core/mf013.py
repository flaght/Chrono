import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf013(mainInflow, mainBuyOrd,  inflowS, inflowM, buyOrdS, buyOrdM, window=10, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    main_avg_ord = mainInflow / mainBuyOrd 
    smain_avg_ord = (inflowS + inflowM) / (buyOrdS + buyOrdM)
    alpha = roller_mean(main_avg_ord / smain_avg_ord, window, 1, method)
    return alpha

    
