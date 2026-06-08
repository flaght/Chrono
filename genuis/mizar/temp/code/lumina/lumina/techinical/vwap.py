# -*- encoding:utf-8 -*-
"""
    计算vwap模块
"""
import numpy as np
import pandas as pd
from ultron.ump.core.helper import pd_rolling_mean,pd_ewm_mean
from ultron.ump.technical.vwap import calc_vwap

def calc_rank(kl_pd):
    close = kl_pd['close']
    vwap_price = calc_vwap(close)
    return vwap_price