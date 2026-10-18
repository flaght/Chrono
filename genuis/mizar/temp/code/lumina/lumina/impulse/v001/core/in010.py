import pdb
import pandas as pd
from lumina.impulse.fixed import *


## adx
def in010(close, high, low, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).to_frame()
    true_range.columns = tr1.columns
    # 计算+DI和-DI
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = roller_mean(true_range, weriod, 1, method)
    plug_dm = roller_mean(plus_dm, weriod, 1, method)
    minus_dm = roller_mean(minus_dm, weriod, 1, method)

    plus_di = 100 * (plug_dm / tr)
    minus_di = -100 * (minus_dm / tr)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100

    adx = roller_mean(dx, weriod, 1, method)

    alpha = roller_mean(adx, window, 1, method)
    return alpha
