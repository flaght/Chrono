# -*- encoding:utf-8 -*-
"""
    计算线liquidity1模块
"""
from ultron.ump.technical.line import Line
import numpy as np
import pdb
def calc_liquidity2(kl_pd, xd, ewm=False):
    vol = kl_pd['volume'].copy()
    vol[vol <= 0] = np.nan
    log_vol = np.log(vol)

    if ewm:
        vol_ma = -log_vol.ewm(span=xd, min_periods=xd).std()
    else:
        vol_ma = -log_vol.rolling(window=xd, min_periods=xd).std()
    line = Line(vol_ma, 'liquidity2')
    return line