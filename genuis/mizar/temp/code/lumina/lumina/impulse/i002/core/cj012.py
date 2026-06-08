import pdb
from lumina.impulse.fixed import *


def cj012(close, volume, window, weriod, ewm=False):
    '''
    量价相关性 
    成交量
    '''
    method = 'ewm' if ewm else 'rolling'
    corr1 = roller_corr(close, volume, weriod, weriod, method)

    alpha = roller_mean(corr1, window, window, method)

    return alpha
