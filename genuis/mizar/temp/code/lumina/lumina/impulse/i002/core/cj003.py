from lumina.impulse.fixed import *

def cj003(close, volume, window, weriod, ewm=False):
    '''
    非流动性变异系数
    非流动性（标准差/平均值）
    '''
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    core1 = rets.div(volume / 10e6)
    core1 = core1.replace([np.inf, -np.inf], np.nan)

    std = roller_std(core1, weriod, weriod, method)
    mean = roller_mean(core1, weriod, weriod, method)

    alpha = std.div(mean)

    alpha = roller_mean(alpha, window, window, method)

    return alpha
