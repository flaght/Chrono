import pdb
import pandas as pd
from lumina.impulse.fixed import *


## 勇攀高峰
def dv004(high, open, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ret = safe_log(close, 1)

    sum1_high = roller_sum(high, weriod, 1, method)
    sum2_high = roller_sum(pow(high, 2), weriod, 1, method)

    sum1_open = roller_sum(open, weriod, 1, method)
    sum2_open = roller_sum(pow(open, 2), weriod, 1, method)

    sum1_low = roller_sum(low, weriod, 1, method)
    sum2_low = roller_sum(pow(low, 2), weriod, 1, method)

    sum1_close = roller_sum(close, weriod, 1, method)
    sum2_close = roller_sum(pow(close, 2), weriod, 1, method)

    mean1 = roller_mean(sum1_high + sum1_open + sum1_low + sum1_close, weriod,
                        1, method)

    std1 = np.sqrt((sum2_high + sum2_open + sum2_low + sum2_close) /
                   (high + open + low + close) -
                   np.power((sum1_high + sum1_open + sum1_low + sum1_close) /
                            (high + open + low + close), 2))

    better = std1 / mean1
    volat = ret / (std1 / mean1)

    rets_flag = np.where(
        better > roller_mean(better, weriod, 1, method) +
        roller_std(better, weriod, 1, method), 1, np.nan)

    higher = better * rets_flag
    alpha = roller_cov(higher, volat, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    ### 填充缺失值， 可能会在指定周期内出现缺失值
    alpha = alpha.fillna(method='ffill').fillna(0)
    return alpha
