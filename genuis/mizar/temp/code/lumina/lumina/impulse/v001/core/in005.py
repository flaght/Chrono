import pdb
import pandas as pd
from lumina.impulse.fixed import *


## macd
def in005(close, window, fast, slow, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ema_fast = roller_mean(close, fast, 1, method)
    ema_slow = roller_mean(close, slow, 1, method)

    macd = ema_fast - ema_slow

    signal = roller_mean(macd, weriod, 1, method)

    hist = macd - signal

    macd = roller_mean(macd, window, 1, method)
    signal = roller_mean(signal, window, 1, method)
    hist = roller_mean(hist, window, 1, method)
    return macd, signal, hist
