import pdb
import pandas as pd
from lumina.impulse.a001.core.base import *


def al001(close, window, weriod, iret=None, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)

    std1 = roller_std(chg, weriod, 1, method)
    avg_std = roller_mean(std1, weriod, 1, method)
    
    risk = avg_std - std1

    ## 计算收益率
    mean1 = roller_mean(chg, weriod, 1, method)

    ## 计算全市场收益
    avg_mean = mean1.mean(axis=1) if iret is None else iret

    risk_values = mean1.sub(avg_mean, axis='rows') * risk

    factor = calc_umr(risk_values, weriod)

    alpha = roller_mean(factor, window, 1, method)

    return alpha
