import pdb
import pandas as pd
from lumina.impulse.fixed import *


##成交额占比熵
def dv001(value, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    aprop = value / roller_sum(value, weriod, 1, method)
    alpha = roller_sum(-aprop * safe_log(aprop, 1), window, 1, method)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha
