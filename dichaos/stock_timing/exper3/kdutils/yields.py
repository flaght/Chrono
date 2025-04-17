import numpy as np
import pandas as pd
import pdb


# 数据转化为矩阵
def data_convert(in_data):
    in_data = in_data.drop_duplicates(subset=['trade_date', 'code'])

    out_data = in_data.pivot_table(index='trade_date',
                                   columns='code',
                                   values=in_data.columns[-1])
    return out_data


def transform(market_data):
    closep = data_convert(market_data[['trade_date', 'code', 'close']].copy())

    openp = data_convert(market_data[['trade_date', 'code', 'open']].copy())

    vol = data_convert(market_data[['trade_date', 'code', 'volume']].copy())

    val = data_convert(market_data[['trade_date', 'code', 'value']].copy())

    closep[vol == 0] = np.nan
    closep.fillna(method='pad', inplace=True)
    closep[closep <= 0] = np.nan
    openp[vol == 0] = np.nan
    openp.fillna(method='pad', inplace=True)
    openp[openp <= 0] = np.nan
    vwap = val / vol
    vwap[np.isinf(vwap)] = np.nan
    pre_vwap = vwap.shift()
    pre_closep = closep.shift()
    pre_openp = openp.shift()
    return closep, openp, vwap, pre_vwap, pre_closep, pre_openp, vol, val


def returns(closep, openp, vwap, pre_vwap, pre_closep, pre_openp, vol):
    ret = np.log((closep) / pre_closep)
    tag = ((vol == 0) | (vol.isna())) & (ret == 0)
    ret[tag] = np.nan

    ret_o2o = np.log((openp) / pre_openp)
    ret_o2o[tag] = np.nan
    ret_c2o = np.log((openp) / pre_closep)
    ret_c2o[tag] = np.nan
    ret_o2c = np.log(closep / openp)
    ret_o2c[tag] = np.nan
    ret_vwap = np.log((vwap) / pre_vwap)
    ret_vwap[tag] = np.nan
    ret_f1r_vv = ret_vwap.shift(-2)
    ret_f1r_cc = ret.shift(-1)
    ret_f1r_oo = ret_o2o.shift(-2)

    total_ret = pd.concat([
        ret.unstack(),
        ret_vwap.unstack(),
        ret_f1r_vv.unstack(),
        ret_f1r_cc.unstack(),
        ret_f1r_oo.unstack(),
        ret_o2o.unstack(),
        ret_c2o.unstack(),
        ret_o2c.unstack()
    ],
                          axis=1)
    total_ret.columns = [
        'ret', 'ret_vwap', 'ret_f1r_vv', 'ret_f1r_cc', 'ret_f1r_oo', 'ret_o2o',
        'ret_c2o', 'ret_o2c'
    ]
    total_ret.reset_index(inplace=True)
    return total_ret


def fetch_returns(market_data):
    closep, openp, vwap, pre_vwap, pre_closep, pre_openp, vol, val = transform(
        market_data)
    return returns(closep, openp, vwap, pre_vwap, pre_closep, pre_openp, vol)
