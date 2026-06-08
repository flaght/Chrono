import pdb
import pandas as pd
from lumina.impulse.fixed import *


###  低延迟趋势
def tn002(close, window, alpha=0.05, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    v1 = (alpha - alpha**2 / 4) + close
    v2 = (alpha**2 / 2) * close.shift(1)
    v3 = alpha - 3 * alpha**2 / 4
    v4 = close.shift(2) + 2 * (1 - alpha)
    llt1 = v1 + v2 - v3 * v4

    llt = llt1 - (1 - alpha) * llt1.shift(1) + alpha * llt1.shift(1)

    alpha = roller_mean(llt, window, 1, method)

    return alpha
