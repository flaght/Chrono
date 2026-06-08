
import numpy as np
import pandas as pd
import bottleneck as bn

def rolling_window(X, window):
    """
    返回2D array的滑窗array的array
    """
    
    pad_width = [(0, 0)] * X.ndim    
    pad_width[0] = (window - 1, 0)    
    X = np.pad(X, pad_width, mode='constant', constant_values=np.nan)    
    
    shape = (X.shape[0] - window +1, window, X.shape[-1])
    strides = (X.strides[0],) + X.strides
    a_rolling = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides, writeable=False)

    return a_rolling

def rolling_mean(X, window):
    """求滚动均值

    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): . Defaults to 0.

    Returns:
        _type_: np.ndarray
    """

    return bn.move_mean(X, window=window, axis=0)

def rolling_std(X, window):
    """求滚动标准差

    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """

    return bn.move_std(X, window=window, axis=0)

def rolling_median(X, window):
    """求滚动中位数

    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """

    return bn.move_median(X, window=window, axis=0)

def rolling_max(X, window):
    """求滚动最大值

    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """

    return bn.move_max(X, window=window, axis=0)

def rolling_min(X, window):
    """求滚动最小值

    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """
    
    return bn.move_min(X, window=window, axis=0)

def rolling_ptp(X, window):
    """求滚动最大值减最小值

    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """

    return bn.move_max(X, window=window, axis=0) - bn.move_min(X, window=window, axis=0)

def rolling_perse(X, window):
    """求滚动百分比：当前值/滚动最大值

    Args:
        X (_type_): np.ndarray
        window (_type_): int

    Returns:
        _type_: np.ndarray
    """
    
    return X / rolling_max(X, window)

def rolling_perser(X, window):
    """求滚动百分比：当前值/滚动最小值

    Args:
        X (_type_): np.ndarray
        window (_type_): int

    Returns:
        _type_: np.ndarray
    """
    
    return X / rolling_min(X, window)

def rolling_quantile(X, window, q=0.8):
    """求滚动分位数

    Args:
        X (_type_): np.ndarray
        window (_type_): int
        q (_type_): float
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """

    return np.quantile(rolling_window(X, window), q, axis=1)

def rolling_sum(X, window):
    """求滚动和

    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """

    return bn.move_sum(X, window=window, axis=0)

def rolling_var(X, window):
    """求滚动方差
    
    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """

    return bn.move_var(X, window=window, axis=0)
    
def rolling_skewness(X, window):
    """求滚动偏度
    
    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """
    return pd.DataFrame(X).rolling(window=window).skew().values

def rolling_kurtosis(X, window):
    """求滚动峰度
    
    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """
    return pd.DataFrame(X).rolling(window=window).kurt().values
    
def rolling_sqrtsum(X, window):
    """求滚动平方求和

    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """
    
    return bn.move_sum(X**2, window=window, axis=0)

def rolling_abssum(X, window):
    """求滚动绝对值求和

    Args:
        X (_type_): np.ndarray
        window (_type_): int
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: np.ndarray
    """
    return bn.move_sum(abs(X), window=window, axis=0)

def rolling_argmax(X, window):
    """求滚动最大值的距离当前的日期数：包含0值

    Args:
        X (_type_): np.ndarray
        window (_type_): int

    Returns:
        _type_: np.ndarray
    """
    
    return bn.move_argmax(X, window=window, axis=0)

def rolling_argmin(X, window):
    """求滚动最小值的距离当前的日期数:包含0值

    Args:
        X (_type_): np.ndarray
        window (_type_): int

    Returns:
        _type_: np.ndarray
    """
    
    return bn.move_argmin(X, window=window, axis=0)

def rolling_rank(X, window):
    """求滚动排序：最小值为-1，最大值为1

    Args:
        X (_type_): np.ndarray
        window (_type_): int

    Returns:
        _type_: np.ndarray
    """
    
    return bn.move_rank(X, window=window, axis=0)

def rolling_zscore(X, window):
    """求滚动zscore

    Args:
        X (_type_): np.ndarray
        window (_type_): int

    Returns:
        _type_: np.ndarray
    """
    
    return (X - rolling_mean(X, window)) / (rolling_std(X, window) + 1e-16)

def rolling_transfrom(X,window=20,method='mean'):
    """求指定方法的滚动变换
    Args:
        X (_type_): np.ndarray
        window (_type_): int

    Returns:
        _type_: np.ndarray
    """
    res = None
    if method == 'mean':
        res = rolling_mean(X,window)
    elif method =='std':
        res = rolling_std(X,window)
    elif method =='sum':
        res = rolling_sum(X,window)
    elif method =='var':
        res = rolling_var(X,window)
    elif method =='median':
        res =  rolling_median(X,window)
    elif method =='max':
        res =  rolling_max(X,window)
    elif method =='min':
        res = rolling_min(X,window)
    elif method =='ptp':
        res = rolling_ptp(X,window)
    elif method =='perse':
        res = rolling_perse(X,window)
    elif method =='perser':
        res = rolling_perser(X,window)
    elif method=='qupper':
        res = rolling_quantile(X,window,q=0.8)
    elif method=='qlower':
        res = rolling_quantile(X,window,q=0.2)
    elif method=='skewness':
        res = rolling_skewness(X,window)
    elif method=='kurtosis':
        res = rolling_kurtosis(X,window)
    elif method=='sqrtsum':
        res = rolling_sqrtsum(X,window)
    elif method=='abssum':
        res = rolling_abssum(X,window)
    elif method=='argmax':
        res = rolling_argmax(X,window)
    elif method=='argmin':
        res = rolling_argmin(X,window)
    elif method=='rank':
        res = rolling_rank(X,window)
    elif method=='zscore':
        res = rolling_zscore(X,window)
    else:
        raise ValueError('method must be in [mean,std,sum,var,median,max,min,ptp,perse,\
                            perser,q_upper,q_lower,skewness,kurtosis,sqrtsum,abssum,argmax,\
                            argmin,rank,zscore]')
        
    return res
    
def rolling_cov(X,Y,window=20):
    """求滚动协方差：简单协方差计算

    Args:
        X (_type_): np.ndarray
        Y (_type_): np.ndarray
        window (_type_): int

    Returns:
        _type_: np.ndarray
    """
    
    x1 = X - rolling_mean(X, window)
    y1 = Y - rolling_mean(Y, window)
    return rolling_sum(x1*y1,window) /window

def rolling_corr(X,Y,window=20):
    """求滚动相关系数：简单相关系数计算

    Args:
        X (_type_): np.ndarray
        X (_type_): np.ndarray
        window (_type_): int

    Returns:
        _type_: np.ndarray
    """
    
    x1 = X - rolling_mean(X, window)
    y1 = Y - rolling_mean(Y, window)
    return rolling_sum(x1*y1,window) /np.sqrt(rolling_sum(x1**2,window)*rolling_sum(y1**2,window))


def rolling_kendalltau(X,Y,window=20):
    """求滚动kendalltau系数：简单kendalltau系数计算

    Args:
        X (_type_): np.ndarray
        Y (_type_): np.ndarray
        window (_type_): int

    Returns:
        _type_: np.ndarray
    """
    
    x1 = X - rolling_mean(X, window)
    y1 = Y - rolling_mean(Y, window)
    return rolling_sum(np.sign(x1)*np.sign(y1),window) /window

def rolling_spearmanr(X,Y,window=20):
    """求滚动spearmanr系数：简单spearmanr系数计算

    Args:
        X (_type_): np.ndarray
        Y (_type_): np.ndarray
        window (_type_): int

    Returns:
        _type_: np.ndarray
    """
    X = rolling_rank(X,window)
    Y = rolling_rank(Y,window)
    
    x1 = X - rolling_mean(X, window)
    y1 = Y - rolling_mean(Y, window)
    return rolling_sum(x1*y1,window) /np.sqrt(rolling_sum(x1**2,window)*rolling_sum(y1**2,window))