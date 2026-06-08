import pdb
from lumina.impulse.fixed import *

def xy003(volume, window, weriod, ewm=False):
    '''
    cumsumvol_mean
    累计成交量均值
    '''

    method = 'ewm' if ewm else 'rolling'
    sum1 = roller_sum(volume, weriod, weriod, method)
    alpha = roller_mean(sum1, window, window, method)
    return alpha