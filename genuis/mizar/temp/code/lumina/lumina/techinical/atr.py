# -*- encoding:utf-8 -*-
"""
    计算线atr模块
"""
import numpy as np
import pandas as pd
from ultron.ump.technical.atr import calc_atr_std


def calc_atr(kl_pd, xd, ewm=True):
    atr = calc_atr_std(kl_pd, xd=xd, ewm=ewm)
    return atr