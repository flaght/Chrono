import pdb
from lumina.impulse.fixed import *


def cj011(close, window, weriod, ewm=False):
    '''
    价格波动率
    以收盘价日内波动率作为指标
    '''
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)

    square = (rets - roller_mean(rets, weriod, weriod, method))**2
    upper_square = square.mask(rets < roller_mean(rets, weriod, weriod, method))
    square = roller_sum(square, weriod, 1, method)
    upper_square = roller_sum(upper_square, weriod, 1, method)
    alpha = upper_square.div(square, axis='rows')

    alpha = roller_mean(alpha, window, window, method)
    return alpha