from lumina.impulse.fixed import *



def oi021(openint, window, weriod, ewm=False):
    '''
    logvol_10tail 
    对数持仓量厚尾分布：10%分位数以下占比
    '''
    

    method = 'ewm' if ewm else 'rolling'
    mask = openint.mask(openint > roller_quantile(openint, 0.1, weriod, 1, 'rolling'))

    need = openint
    need = need.mask(mask.isna())
    need = roller_sum(need, weriod, 1, method)

    alpha = need.div(roller_sum(openint, weriod, weriod, method))

    alpha = roller_mean(alpha, window, int(window), method)
    return alpha
