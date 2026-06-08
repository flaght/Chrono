import pdb
from lumina.impulse.fixed import *


def cj007(close, high, low, window, weriod, ewm=False):
    '''
    高价振幅
    价格最高（20%）部分平均振幅
    '''
    method = 'ewm' if ewm else 'rolling'
    mask = close.mask(
        close.sub(roller_quantile(close, 0.8, weriod, 1, 'rolling'),
                          axis='rows') < 0)
    
    core1 = high - low
    core1 = core1.mask(mask.isna())
    core1 = core1.fillna(0) 
    alpha = roller_sum(core1, weriod, 1, method)
    alpha = roller_mean(alpha, window, 1, method)
    
    return alpha