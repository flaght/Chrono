import pdb
from lumina.impulse.fixed import *


## tcd
def iv007(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)

    chg_up = chg.mask(chg <= 0, 0)
    chg_up_sum = roller_sum(chg_up, weriod, 1, method)
    up1 = chg_up / chg_up_sum

    chg_down = chg.mask(chg >= 0, 0).abs()
    chg_down_sum = roller_sum(chg_down, weriod, 1, method)
    down1 = chg_down / chg_down_sum

    factor = up1 - down1

    alpha = roller_mean(factor, window, 1, method)

    return alpha
