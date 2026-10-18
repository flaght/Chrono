from lumina.impulse.fixed import *


def oi012(openint, window, weriod, ewm=False):
    '''
    持仓占比熵
    以持仓量占比为 p，带入熵公式
    '''
    method = 'ewm' if ewm else 'rolling'
    sum = roller_sum(openint, weriod, weriod, method)
    ratio = openint.div(sum)

    ratio_entropy = ratio.mul(np.log(ratio))

    alpha = roller_sum(ratio_entropy, weriod, weriod, method)

    alpha = roller_mean(alpha, window, window, method)

    return alpha
