from lumina.impulse.fixed import *

def oi018(openint, window, weriod, ewm=False):
    '''
    cumsumvol_std
    累计持仓量标准差
    '''
    method = 'ewm' if ewm else 'rolling'
    sum1 = roller_sum(openint, weriod, weriod, method)
    std1 = roller_std(sum1, window, window, method)

    alpha = roller_mean(std1, window, window, method)

    return alpha