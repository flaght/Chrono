import pdb
from lumina.impulse.fixed import *


# ui
def tv014(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    highest_close = roller_max(close, weriod, 1, 'rolling')
    downside = 100 * (close - highest_close) / highest_close
    d2 = downside**2

    ui = roller_mean(d2, weriod, 1, method)

    alpha = roller_mean(ui, window, 1, method)

    return alpha
