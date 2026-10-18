import pdb
from lumina.impulse.fixed import *

def xy002(volume, window, weriod, ewm=False):
    '''
    cumsumvol_std
    累计成交量标准差
    '''
    method = 'ewm' if ewm else 'rolling'
    sum1 = roller_sum(volume, weriod, weriod, method)
    std1 = roller_std(sum1, window, window, method)

    alpha = roller_mean(std1, window, window, method)

    return alpha