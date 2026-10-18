import pdb
import numpy as np
from lumina.impulse.fixed import *


#upward
def ixy004(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    rets_up = rets.where(rets > 0, 0)
    rets_std = roller_std(rets, weriod, 1, method)
    rets_up_std = roller_std(rets_up, weriod, 1, method)
    alpha = (rets_up_std / rets_std) * np.sqrt(252)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha
