# -*- encoding:utf-8 -*-
"""
    计算线ICHIMOKU模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_ichimoku(kl_pd, tenkan, kijun, senkou, ewm=True):
    tenkan = int(tenkan) if tenkan and tenkan > 0 else 9
    kijun = int(kijun) if kijun and kijun > 0 else 26
    senkou = int(senkou) if senkou and senkou > 0 else 52

    high = kl_pd['high']
    low = kl_pd['low']

    if ewm:
        tenkan_lowest_low = pd_ewm_mean(low, span=tenkan, min_periods=tenkan)
        tenkan_highest_high = pd_ewm_mean(high,
                                          span=tenkan,
                                          min_periods=tenkan)
        tenkan_sen = (tenkan_lowest_low + tenkan_highest_high) / 2

        kijun_lowest_low = pd_ewm_mean(low, span=kijun, min_periods=kijun)
        kijun_highest_high = pd_ewm_mean(high, span=kijun, min_periods=kijun)
        kijun_sen = (kijun_lowest_low + kijun_highest_high) / 2

        senkou_lowest_low = pd_ewm_mean(low, span=senkou, min_periods=senkou)
        senkou_highest_high = pd_ewm_mean(high,
                                          span=senkou,
                                          min_periods=senkou)
        senkou_sen = (senkou_lowest_low + senkou_highest_high) / 2

        span_a = 0.5 * (tenkan_sen + kijun_sen)
        span_b = senkou_sen
    else:
        tenkan_lowest_low = pd_rolling_mean(low,
                                            window=tenkan,
                                            min_periods=tenkan)
        tenkan_highest_high = pd_rolling_mean(high,
                                              window=tenkan,
                                              min_periods=tenkan)
        tenkan_sen = (tenkan_lowest_low + tenkan_highest_high) / 2

        kijun_lowest_low = pd_rolling_mean(low,
                                           window=kijun,
                                           min_periods=kijun)
        kijun_highest_high = pd_rolling_mean(high,
                                             window=kijun,
                                             min_periods=kijun)
        kijun_sen = (kijun_lowest_low + kijun_highest_high) / 2

        senkou_lowest_low = pd_rolling_mean(low,
                                            window=senkou,
                                            min_periods=senkou)
        senkou_highest_high = pd_rolling_mean(high,
                                              window=senkou,
                                              min_periods=senkou)
        senkou_sen = (senkou_lowest_low + senkou_highest_high) / 2

        span_a = 0.5 * (tenkan_sen + kijun_sen)
        span_b = senkou_sen

    span_a = pd.Series(span_a).fillna(method='bfill')
    span_b = pd.Series(span_b).fillna(method='bfill')

    return Line(span_a, 'tenkan_sen'), Line(span_b, 'kijun_sen')
