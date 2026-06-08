import pdb
from lumina.impulse.fixed import *


def cj009(close, window, weriod, ewm=False):
    '''
    价格波动率
    以收盘价日内波动率作为指标
    '''
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)

    mean = roller_mean(rets, weriod, weriod, method)

    alpha = roller_max(rets.sub(mean).abs(), weriod, weriod, 'rolling')

    alpha = roller_mean(alpha, window, window, method)

    return alpha
