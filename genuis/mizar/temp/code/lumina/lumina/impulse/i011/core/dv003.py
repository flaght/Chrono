import pdb
import pandas as pd
from lumina.impulse.fixed import *

## 灾后重建
def dv003(high, open, low, close, window, weriod, ewm=False):
    
    method = 'ewm' if ewm else 'rolling'

    ret = safe_log(close, 1)

    mhigh = roller_sum(high, weriod, 1, method)
    mopen = roller_sum(open, weriod, 1, method)
    mlow = roller_sum(low, weriod, 1, method)
    mclose = roller_sum(close, weriod, 1, method)

    phigh = roller_sum(pow(high, 2), weriod, 1, method)
    popen = roller_sum(pow(open, 2), weriod, 1, method)
    plow = roller_sum(pow(low, 2), weriod, 1, method)
    pclose = roller_sum(pow(close, 2), weriod, 1, method)

    mean1 = (mhigh + mopen + mlow + mclose) / (phigh + popen + plow + pclose)
    std1 = np.sqrt((phigh + popen + plow + pclose) /
                   (high + open + low + close) -
                   np.power((mhigh + mopen + mlow + mclose) /
                            (high + open + low + close), 2))
    better = std1 / mean1
    volat = ret / (std1 / mean1)

    alpha = roller_cov(better, volat, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha
