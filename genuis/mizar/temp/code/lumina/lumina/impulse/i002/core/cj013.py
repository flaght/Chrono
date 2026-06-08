import pdb
from lumina.impulse.fixed import *


def cj013(close, volume, window, weriod, ewm=False):
    '''
    加权偏度 
    成交量加权收盘价偏度
    '''
    method = 'ewm' if ewm else 'rolling'
    sum1 = roller_sum(volume, weriod, weriod, method)
    ratio = volume.div(sum1)

    weight1 = close.mul(ratio)

    alpha = roller_skew(weight1, weriod, weriod, method)
    alpha = roller_mean(alpha, window, window, method)
    return alpha