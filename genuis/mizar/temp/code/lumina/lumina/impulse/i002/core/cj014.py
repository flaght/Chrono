import pdb
from lumina.impulse.fixed import *


def cj014(open, high, low, close, vwap, window, weriod, ewm=False):
    '''
    时量价比 
    时间加权价格/量加权价格
    '''
    method = 'ewm' if ewm else 'rolling'
    twap = (open + high + low + close) / 4
    twap = roller_mean(twap, weriod, weriod, method)
    vwap = roller_mean(vwap, weriod, weriod, method)
    alpha = twap.div(vwap + 1e-5, axis='rows')
    alpha = roller_mean(alpha, window, window, method)

    return alpha