import pdb
import pandas as pd
from lumina.impulse.fixed import *


## RSI
def ht003(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    delta = close.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = roller_mean(gain, weriod, 1, method)
    avg_loss = roller_mean(loss, weriod, 1, method)

    rs = avg_gain / avg_loss + 1e-6
    alpha = 100 - (100 / (1 + rs))

    alpha = roller_mean(alpha, window, 1, method)

    return alpha
