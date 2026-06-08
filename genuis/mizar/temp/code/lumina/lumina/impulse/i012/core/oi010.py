from lumina.impulse.fixed import *


def oi010(openint, window, weriod, ewm=False):
    '''
    波峰
    以日内 1 分钟成交量 k 线数据均值+1 倍标准差作为
    峰值筛选，以局部峰值筛选后的局部峰值 k 线数量
    '''
    method = 'ewm' if ewm else 'rolling'
    oit = openint / 1e6
    mean = roller_mean(oit, weriod, weriod, method)
    std = roller_std(oit, weriod, weriod, method)

    alpha = oit.add(np.where(oit > (mean + std), 1, 0))
    alpha = roller_sum(alpha, weriod, weriod, method)
    alpha = roller_mean(alpha, window, window, method)
    return alpha
