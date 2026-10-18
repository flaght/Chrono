# -*- encoding:utf-8 -*-
"""
    计算wave模块
"""
import numpy as np
import pandas as pd
from ultron.ump.core.helper import pd_rolling_mean,pd_ewm_mean
from ultron.ump.technical.wave import calc_wave_std

def calc_wave(kl_pd, xd):
    close = kl_pd['close']
    vwap_price = calc_wave_std(close)
    return vwap_price