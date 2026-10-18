import pdb
import pandas as pd
from lumina.impulse.fixed import *


def slopes(values):
    y = np.nan_to_num(values)
    row_indices = np.arange(y.shape[0])
    x = np.tile(row_indices[:, np.newaxis], (1, y.shape[1]))
    X = np.hstack((np.ones(
        (x.shape[0], 1)), x))  # add constant X = sm.add_constant(X)
    slopes = np.linalg.lstsq(X, y, rcond=None)[0]
    slope = np.median(slopes, axis=0)
    intercepts = y - slope * x
    intercept = np.median(intercepts, axis=0)
    return intercept + slope * (x[-1] + 1)


def tn007(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ## ols回归
    x1_rolling = rolling_window(close.values, window=weriod)

    icu = pd.DataFrame(map(lambda x1: slopes(x1), x1_rolling),
                         index=close.index,
                         columns=close.columns)
    
    alpha = roller_mean(icu, window, 1, method)
    return alpha
