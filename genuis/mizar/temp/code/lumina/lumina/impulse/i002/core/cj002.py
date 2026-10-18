import pdb
from lumina.impulse.fixed import *


def cj002(volume, window, weriod, ewm=False):
    '''
    波峰
    以日内 1 分钟成交量 k 线数据均值+1 倍标准差作为
    峰值筛选，以局部峰值筛选后的局部峰值 k 线数量
    '''
    method = 'ewm' if ewm else 'rolling'
    mean = roller_mean(volume, weriod, weriod, method)
    std = roller_std(volume, weriod, weriod, method)

    alpha = volume.add(np.where(volume > (mean + std), 1, 0))
    alpha = roller_sum(alpha, weriod, weriod, method)
    alpha = roller_mean(alpha, window, window, method)
    return alpha
