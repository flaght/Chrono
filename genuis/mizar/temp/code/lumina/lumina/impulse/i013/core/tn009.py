import pdb
import pandas as pd
from lumina.impulse.fixed import *


def wls_beta(values, target, weight):
    y = np.nan_to_num(values)
    X = np.nan_to_num(target)
    weight = np.nan_to_num(weight)
    X = X * weight
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    beta = np.linalg.lstsq(X, y * weight, rcond=None)[0][1]
    return beta

### 成交量加权RSRS
def tn009(low, high, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    weight = volume / roller_sum(volume, weriod, 1, method)

    ## wls回归
    x1_rolling = rolling_window(low.values, window=weriod)
    x2_rolling = rolling_window(high.values, window=weriod)
    x3_rolling = rolling_window(weight.values, window=weriod)

    wbeta = pd.DataFrame(map(lambda x1, x2, x3: wls_beta(x1, x2, x3),
                             x1_rolling, x2_rolling, x3_rolling),
                         index=low.index,
                         columns=low.columns)
    wcorr = roller_corr(low * weight, high * weight, weriod, 1, method)**2

    ### 原始RSRS beta系数
    rsrs1 = wbeta
    ## 标准分RSRS: 原始RSS标准化
    rsrs2 = (rsrs1 - roller_mean(rsrs1, weriod, 1, method)) / roller_std(
        rsrs1, weriod, 1, method)

    ## 修正标准分RSRS:标准分RSRS乘以滚动相关性
    rsrs3 = rsrs2 * wcorr

    ### 右偏标准分RSRS:修正标准分RSRS * 原始RSRS
    rsrs4 = rsrs3 * rsrs1

    #### 钝化 RSRS: 标准分RSRS * 滚动相关性 ** (2 * 收益率分位数)
    #rss4 = rss1 * rss3

    alpha1 = roller_mean(rsrs1, window, 1, method)
    alpha2 = roller_mean(rsrs2, window, 1, method)
    alpha3 = roller_mean(rsrs3, window, 1, method)
    alpha4 = roller_mean(rsrs4, window, 1, method)

    return alpha1, alpha2, alpha3, alpha4
