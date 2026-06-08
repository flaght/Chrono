from lumina.impulse.fixed import *


def oi015(close, openint, window, weriod, ewm=False):
    '''
    高量交易成本 
    最高20%成交区间价格偏离
    '''
    method = 'ewm' if ewm else 'rolling'
    mask = openint.mask(openint < roller_quantile(openint, 0.8, weriod, 1, 'rolling'))

    mean = roller_mean(close, weriod, weriod, method)

    need = close.sub(mean)
    need = need.mask(mask.isna())

    alpha = roller_mean(need, weriod, 1, method)

    alpha = roller_mean(alpha, window, int(window / 2), method)

    return alpha
