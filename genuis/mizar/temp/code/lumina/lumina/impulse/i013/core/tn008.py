import pdb
import pandas as pd
from lumina.impulse.fixed import *


def ols_beta(values, target):
    y = np.nan_to_num(values)
    x = np.nan_to_num(target)
    X = np.hstack((np.ones(
        (x.shape[0], 1)), x)) 
    beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
    return beta


## RSRS 
def tn008(low, high, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ## ols回归
    x1_rolling = rolling_window(low.values, window=weriod)
    x2_rolling = rolling_window(high.values, window=weriod)

    obeta = pd.DataFrame(map(lambda x1, x2: ols_beta(x1, x2), x1_rolling,
                             x2_rolling),
                         index=low.index,
                         columns=low.columns)
    ## 滚动相关性
    ocorr = roller_corr(low, high, weriod, 1, method)**2

    ### 原始RSRS beta系数
    rsrs1 = obeta
    ## 标准分RSRS: 原始RSS标准化
    rsrs2 = (rsrs1 - roller_mean(rsrs1, weriod, 1, method)) / roller_std(
        rsrs1, weriod, 1, method)
    
    ## 修正标准分RSRS:标准分RSRS乘以滚动相关性
    rsrs3 = rsrs2 * ocorr

    ### 右偏标准分RSRS:修正标准分RSRS * 原始RSRS
    rsrs4 = rsrs3 * rsrs1

    #### 钝化 RSRS: 标准分RSRS * 滚动相关性 ** (2 * 收益率分位数)
    #rss4 = rss1 * rss3

    alpha1 = roller_mean(rsrs1, window, 1, method)
    alpha2 = roller_mean(rsrs2, window, 1, method)
    alpha3 = roller_mean(rsrs3, window, 1, method)
    alpha4 = roller_mean(rsrs4, window, 1, method)

    return alpha1, alpha2, alpha3, alpha4