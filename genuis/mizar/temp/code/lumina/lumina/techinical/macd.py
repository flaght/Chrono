# -*- encoding:utf-8 -*-
"""
    计算线macd模块
"""
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean

def calc_macd(kl_pd, fast, slow, ewm=True):
    if ewm:
        fast_ma = pd_ewm_mean(kl_pd['close'], span=fast, min_periods=fast)
        slow_ma = pd_ewm_mean(kl_pd['close'], span=slow, min_periods=slow)
    else:
        fast_ma = pd_rolling_mean(kl_pd['close'], window=fast, min_periods=fast)
        slow_ma = pd_rolling_mean(kl_pd['close'], window=slow, min_periods=slow)
    macd = fast_ma - slow_ma
    line = Line(macd, 'macd')
    return line