import pdb
import pandas as pd
from lumina.impulse.fixed import *

# FOC
def iv010(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)
    factor = roller_corr(chg, chg.shift(1), weriod, 1, method)
    alpha = roller_mean(factor, window, 1, method)

    return alpha
    