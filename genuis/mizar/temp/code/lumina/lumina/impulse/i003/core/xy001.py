import pdb
from lumina.impulse.fixed import *


def xy001(volume, window, weriod, ewm=False):
    '''
    vol_maxmean
    极大值成交量的均值
    '''
    
    method = 'ewm' if ewm else 'rolling'
    core1 = volume.where((volume > roller_quantile(volume, 0.8, weriod, 1, 'rolling')))
    core1 = core1.fillna(0)
    alpha = roller_sum(core1, weriod, 1, method)

    
    alpha = roller_mean(alpha, window, window, method)
    return alpha