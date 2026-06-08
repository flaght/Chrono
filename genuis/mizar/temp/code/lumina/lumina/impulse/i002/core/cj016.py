import pdb
from lumina.impulse.fixed import *


def cj016(close, volume, window, weriod, ewm=False):
    '''
    高价成交占比 
    最高20%价格区间成交量占比
    '''
    method = 'ewm' if ewm else 'rolling'

    mask = close.mask(close < roller_quantile(close, 0.8, weriod, 1, 'rolling'))

    need = volume.copy()
    need = need.mask(mask.isna())
    need = need.fillna(method='ffill')

    final = roller_sum(need, weriod, 1, method)
    alpha = final.div(roller_sum(volume, weriod, weriod, method))

    alpha = roller_mean(alpha, window, int(window / 2), method)
    return alpha
