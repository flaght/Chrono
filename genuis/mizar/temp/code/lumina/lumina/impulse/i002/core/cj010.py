import pdb
from lumina.impulse.fixed import *


def cj010(open, high, low, close, window, weriod, ewm=False):
    '''
    积分相对价格位置
    （时间均价-区间最低价）/（区间最高价-区间最低价）
    '''
    method = 'ewm' if ewm else 'rolling'
    twap = (open + high + low + close) / 4
    alpha = (twap - low).div((high - low) + 1e-5, axis='rows')

    alpha = roller_mean(alpha, weriod, weriod, method)

    alpha = roller_mean(alpha, window, window, method)
    return alpha
