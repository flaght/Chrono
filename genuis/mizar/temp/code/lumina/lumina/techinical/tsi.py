# -*- encoding:utf-8 -*-
"""
    计算线tsi模块
"""
import pdb
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_tsi(kl_pd, fast, slow, xd, scalar, drift, ewm=True):
    if ewm:
        slow_ma = pd_ewm_mean(kl_pd['close'].diff(drift), span=slow, min_periods=slow)
        fast_slow_ma = pd_ewm_mean(slow_ma, span=fast, min_periods=fast)

        abs_slow_ma = pd_ewm_mean(kl_pd['close'].diff(drift).abs(), span=fast, min_periods=fast)
        abs_fast_slow_ma = pd_ewm_mean(abs_slow_ma, span=fast, min_periods=fast)
    else:
        slow_ma = pd_rolling_mean(kl_pd['close'].diff(drift), window=slow, min_periods=slow)
        fast_slow_ma = pd_rolling_mean(slow_ma, window=fast, min_periods=fast)

        abs_slow_ma = pd_rolling_mean(kl_pd['close'].diff(drift), window=slow, min_periods=slow)
        abs_fast_slow_ma = pd_rolling_mean(abs_slow_ma, window=fast, min_periods=fast)

    tsi = scalar * fast_slow_ma / abs_fast_slow_ma

    if ewm:
        tsi = pd_ewm_mean(tsi, span=xd, min_periods=xd)
    else:
        tsi = pd_rolling_mean(tsi, window=xd, min_periods=xd)

    line = Line(tsi, 'tsi')
    return line

    