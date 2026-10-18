from lumina.impulse.fixed import *


def oi019(openint, window, weriod, ewm=False):
    '''
    cumsumvol_mean
    累计持仓量均值
    '''

    method = 'ewm' if ewm else 'rolling'
    sum1 = roller_sum(openint, weriod, weriod, method)
    alpha = roller_mean(sum1, window, window, method)
    return alpha
