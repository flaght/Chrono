from lumina.impulse.fixed import *

def cj006(volume, window, weriod, ewm=False):
    '''
    成交占比熵
    以成交量占比为 p，带入熵公式
    '''
    method = 'ewm' if ewm else 'rolling'
    sum =  roller_sum(volume, weriod, weriod, method)
    ratio = volume.div(sum)

    ratio_entropy = ratio.mul(np.log(ratio))

    alpha = roller_sum(ratio_entropy, weriod, weriod, method)

    alpha = roller_mean(alpha, window, window, method)

    return alpha