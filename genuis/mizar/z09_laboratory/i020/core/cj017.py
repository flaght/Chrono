from lumina.impulse.fixed import *
import numpy as np


def cj017(close, volume, window, weriod, ewm=False):
    '''
    成交量加权波动率调整的动量因子，叠加量价滚动相关性确认
    原始动量信号（20日累计收益）除以成交量加权的收益率标准差，
    再乘以量价相关性确认权重，最后平滑输出
    '''
    method = 'ewm' if ewm else 'rolling'
    
    # 收益率：每日对数收益率
    rets = safe_shift(close, 1)
    
    # 20日累计收益（对数收益）
    momentum = safe_shift(close, drift=20)
    
    # 成交量归一化权重
    volume_mean = roller_mean(volume, weriod, weriod, method)
    volume_weight = safe_div(volume, volume_mean)
    
    # 加权收益率及平方序列
    weighted_ret = volume_weight * rets
    weighted_ret2 = volume_weight * (rets ** 2)
    
    # 滚动求和
    w_ret_sum = roller_sum(weighted_ret, weriod, weriod, method)
    w_ret2_sum = roller_sum(weighted_ret2, weriod, weriod, method)
    w_sum = roller_sum(volume_weight, weriod, weriod, method)
    
    # 加权方差 = E[w*r^2]/E[w] - (E[w*r]/E[w])^2
    mean_wr = safe_div(w_ret_sum, w_sum)
    mean_wr2 = safe_div(w_ret2_sum, w_sum)
    var_w = mean_wr2 - (mean_wr ** 2)
    
    # 成交量加权波动率
    vol_w = np.sqrt(np.maximum(var_w, 0))
    
    # 基础动量除以波动率
    base_alpha = safe_div(momentum, vol_w)
    
    # 滚动相关性：量价相关系数
    corr = roller_corr(close, volume, weriod, weriod, method)
    
    # 确认权重：将相关性从[-1,1]映射到[0,1]
    confirmation_weight = (corr + 1.0) * 0.5
    
    # 最终Alpha = 基础动量 * 确认权重
    alpha = base_alpha * confirmation_weight
    
    # 最终平滑
    alpha = roller_mean(alpha, window, 1, method)
    
    return alpha
