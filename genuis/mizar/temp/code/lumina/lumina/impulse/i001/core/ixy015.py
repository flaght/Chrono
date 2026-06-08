import pdb
from lumina.impulse.fixed import *


#rsj
def ixy015(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)

    rv = rets**2
    rv = roller_sum(rv, weriod, 1, method)

    ##RV_UP
    rv_up = rets.copy()
    rv_up = rv_up.mask(rv_up <= 0, 0)
    rv_up = rv_up**2

    rv_up = roller_sum(rv_up, weriod, 1, method)
    ##RV_DOWN
    rv_down = rets.copy()
    rv_down = rv_down.mask(rv_down >= 0, 0)
    rv_down = rv_down**2

    rv_down = roller_sum(rv_down, weriod, 1, method)

    alpha = (rv_up - rv_down) / rv

    alpha = roller_mean(alpha, window, 1, method)
    return alpha
