from lumina.impulse.fixed import *


## 一致买入交易
def oi045(close, open, high, low, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    if_grow = np.where(close > open, 1, 0)
    if_con = np.where(np.abs(close - open) <= 0.5 * (high - low), 1, 0)

    alpha = roller_sum(openint * if_grow * if_con, weriod, 1,
                       method) / roller_sum(openint, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha
