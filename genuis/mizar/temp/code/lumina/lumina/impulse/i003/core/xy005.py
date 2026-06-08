import pdb
from lumina.impulse.fixed import *


def xy005(volume, window, weriod, ewm=False):
    '''
    logvol_10tail 
    对数成交量厚尾分布：10%分位数以下占比
    '''
    

    method = 'ewm' if ewm else 'rolling'
    mask = volume.mask(volume > roller_quantile(volume, 0.1, weriod, 1, 'rolling'))

    need = volume
    need = need.mask(mask.isna())
    need = need.fillna(0)
    need = roller_sum(need, weriod, 1, method)

    alpha = need.div(roller_sum(volume, weriod, weriod, method))

    alpha = roller_mean(alpha, window, int(window), method)
    return alpha
