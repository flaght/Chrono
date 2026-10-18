from lumina.impulse.fixed import *



def oi013(close, openint, window, weriod, ewm=False):
    '''
    量价相关性 
    持仓量
    '''
    method = 'ewm' if ewm else 'rolling'
    corr1 = roller_corr(close, openint, weriod, weriod, method)

    alpha = roller_mean(corr1, window, window, method)

    return alpha
