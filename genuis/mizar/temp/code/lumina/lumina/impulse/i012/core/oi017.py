from lumina.impulse.fixed import *


def oi017(openint, window, weriod, ewm=False):
    '''
    vol_maxmean
    极大值持仓量的均值
    '''
    method = 'ewm' if ewm else 'rolling'
    core1 = openint.where((openint > roller_quantile(openint, 0.8, weriod, 1, 'rolling')))
    alpha = roller_sum(core1, weriod, 1, method)

    alpha = roller_mean(alpha, window, window, method)
    return alpha
