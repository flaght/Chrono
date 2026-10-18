from lumina.impulse.fixed import *


def oi014(close, openint, window, weriod, ewm=False):
    '''
    加权偏度 
    持仓量加权收盘价偏度
    '''
    method = 'ewm' if ewm else 'rolling'
    sum1 = roller_sum(openint, weriod, weriod, method)
    ratio = openint.div(sum1)

    weight1 = close.mul(ratio)

    alpha = roller_skew(weight1, weriod, weriod, method)
    alpha = roller_mean(alpha, window, window, method)
    return alpha
