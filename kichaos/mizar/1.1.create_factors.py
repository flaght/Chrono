### 创建lumina所有的因子
import datetime
import pdb, os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from alphacopilot.api.calendars import advanceDateByCalendar
from kdutils.macro2 import *
from kdutils.ttimes import get_dates
from kdutils.data import fetch_main_market
import lumina.env as env

env.g_format = 2

import lumina.impulse.i001 as i001
import lumina.impulse.i002 as i002
import lumina.impulse.i003 as i003
import lumina.impulse.i004 as i004
import lumina.impulse.i005 as i005
import lumina.impulse.i006 as i006
import lumina.impulse.i007 as i007
import lumina.impulse.i008 as i008
import lumina.impulse.i009 as i009
import lumina.impulse.i010 as i010
import lumina.impulse.i011 as i011
import lumina.impulse.i012 as i012
import lumina.impulse.i013 as i013


def callback_save(factors_data, name, method, g_instruments, start_date,
                  end_date):
    cond1 = (factors_data.index.get_level_values(
        level=0) >= start_date) & (factors_data.index.get_level_values(
            level=0) <= (datetime.datetime.strptime(end_date, '%Y-%m-%d') +
                         datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    factors_data = factors_data[cond1]
    ff = factors_data.sort_index().reset_index()
    ff1 = ff  #ff.set_index(['trade_time', 'code']).unstack()
    dirs = os.path.join(base_path, method, g_instruments, 'factors')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs,
                            '{0}_factors.feather'.format(name.split('.')[-1]))
    ff1.sort_index().reset_index(drop=True).to_feather(filename)


def calculate_factors(data, callback, method, g_instruments, start_date,
                      end_date):

    def run(data, i00, callback, method, g_instruments, start_date, end_date):
        res = []
        for f in i00.__all__:
            print(f)
            cls = getattr(i00, f)
            obj = cls()
            r1 = obj.calc_impulse(data.copy())
            values = list(r1.values())
            values1 = [v.sort_index() for v in values]
            dt = pd.concat(values1, axis=1).sort_index()
            res.append(dt)
        data = pd.concat(res, axis=1)
        callback(factors_data=data,
                 name=i00.__name__,
                 method=method,
                 g_instruments=g_instruments,
                 start_date=start_date,
                 end_date=end_date)

    run(data, i001, callback, method, g_instruments, start_date, end_date)
    run(data, i002, callback, method, g_instruments, start_date, end_date)
    run(data, i003, callback, method, g_instruments, start_date, end_date)
    run(data, i004, callback, method, g_instruments, start_date, end_date)
    run(data, i005, callback, method, g_instruments, start_date, end_date)
    run(data, i006, callback, method, g_instruments, start_date, end_date)
    run(data, i007, callback, method, g_instruments, start_date, end_date)
    run(data, i008, callback, method, g_instruments, start_date, end_date)
    run(data, i009, callback, method, g_instruments, start_date, end_date)
    run(data, i010, callback, method, g_instruments, start_date, end_date)
    run(data, i011, callback, method, g_instruments, start_date, end_date)
    run(data, i012, callback, method, g_instruments, start_date, end_date)
    run(data, i013, callback, method, g_instruments, start_date, end_date)


def single_factors(method):
    g_instruments = 'rbb'
    start_date, end_date = get_dates(method)
    start_time = advanceDateByCalendar('china.sse', start_date,
                                       '-{0}b'.format(1)).strftime('%Y-%m-%d')
    pdb.set_trace()
    data = fetch_main_market(begin_date=start_time,
                             end_date=end_date,
                             codes=[INSTRUMENTS_CODES[g_instruments]])
    data = data.set_index(['trade_time', 'code']).unstack()
    calculate_factors(data,
                      callback=callback_save,
                      method=method,
                      start_date=start_date,
                      g_instruments=g_instruments,
                      end_date=end_date)


def batch_factors(method):
    start_date, end_date = get_dates(method)
    start_time = advanceDateByCalendar('china.sse', start_date,
                                       '-{0}b'.format(1)).strftime('%Y-%m-%d')
    data = fetch_main_market(begin_date=start_time,
                             end_date=end_date,
                             codes='')
    codes = data['code'].unique().tolist()
    data = data.set_index(['trade_time', 'code']).unstack()
    for code in codes:
        if code not in RINSTRUMENTS_CODES:
            continue
        idx_slice = pd.IndexSlice
        dt = data.loc[:, idx_slice[:, code]]
        calculate_factors(dt,
                          callback=callback_save,
                          method=method,
                          start_date=start_date,
                          g_instruments=RINSTRUMENTS_CODES[code],
                          end_date=end_date)


# TODO: 创建因子
def create_yields(data, horizon, offset=0):
    df = data.copy()
    df.set_index("trade_time", inplace=True)
    ## chg为log收益
    df['nxt1_ret'] = df['chg_pct']
    df = df.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    df = df.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    df.name = 'nxt1_ret'
    return df


def build_yields(g_instruments,
                 method,
                 start_time,
                 end_time,
                 categories,
                 horizon_sets=[]):
    data = fetch_main_market(begin_date=start_time,
                             end_date=end_time,
                             codes=[INSTRUMENTS_CODES[g_instruments]])

    ### 收益率 o2o T+1期开盘价和T+2期开盘价比. T期算因子, T+1期的开盘价交易，T+2期开盘价为一次收益计算
    if categories == 'o2o':
        openp = data.set_index(['trade_time', 'code'])['open'].unstack()
        pre_openp = openp.shift(1)
        ret_o2o = np.log((openp) / pre_openp)
        yields_data = ret_o2o.shift(-2)
        yields_data = yields_data.stack()
        yields_data.name = 'chg_pct'
        yields_data = yields_data.reset_index()
    ##持仓目标收益率
    dirs = os.path.join(base_path, method, g_instruments, 'yields')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    for horizon in horizon_sets:
        df = create_yields(data=yields_data, horizon=horizon)
        filename = os.path.join(dirs,
                                '{0}_{1}h.feather'.format(categories, horizon))
        pdb.set_trace()
        df.reset_index().to_feather(filename)
        print('save yields data to {0}'.format(filename))


# TODO: 创建收益率
def generate_yileds(method, g_instruments):
    start_date, end_date = get_dates(method)
    start_time = advanceDateByCalendar('china.sse', start_date,
                                       '-{0}b'.format(0)).strftime('%Y-%m-%d')
    build_yields(g_instruments=g_instruments,
                 method=method,
                 start_time=start_time,
                 end_time=end_date,
                 categories='o2o',
                 horizon_sets=[1, 3, 5])


# TODO：合并数据 # 1.原始数据 2.因子挖掘数据
def fetch_native_factors(method, g_instruments, names, fillna=True):
    pdb.set_trace()
    dirs = os.path.join(base_path, method, g_instruments, 'factors')
    res = []
    for filename in os.listdir(dirs):
        print(os.path.join(dirs, filename))
        dt = pd.read_feather(os.path.join(dirs, filename))
        dt = dt.set_index(['trade_time', 'code'])
        cols = [col for col in dt.columns if col in names]
        dt1 = dt[cols].sort_index() if len(cols) > 0 else dt
        if not dt1.empty:
            res.append(dt1)
    data = pd.concat(res, axis=1).sort_index()
    if fillna:
        data = data.unstack().fillna(method='ffill')
        data = data.stack().reset_index()
    return data


def fetch_returns(method, g_instruments, types, horizon=1):
    dirs = os.path.join(base_path, method, g_instruments, 'yields')
    filename = os.path.join(dirs, '{0}_{1}h.feather'.format(types, horizon))
    return pd.read_feather(filename)


def merge_data(method,
               g_instruments,
               horizon,
               category,
               fillna=True,
               types='o2o'):
    if category == 1:
        filename = "{}_{}".format("native",
                                          "fillna" if fillna else "nofill")
        factors_data = fetch_native_factors(method=method,
                                            g_instruments=g_instruments,
                                            names=[],
                                            fillna=fillna)
    ret_f1r_oo = fetch_returns(method=method,
                               g_instruments=g_instruments,
                               types=types,
                               horizon=horizon)
    total_data = factors_data.merge(ret_f1r_oo, on=['trade_time', 'code'])
    dirs = os.path.join(base_path, method, g_instruments, 'merged')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(
        dirs, "{0}_{1}h_{2}.feather".format(types, horizon, filename))
    print('save merged data to {0}'.format(filename))
    total_data.to_feather(filename)


if __name__ == "__main__":
    method = 'kimto1'
    g_instruments = 'rbb'
    #single_factors(method=method)
    #generate_yileds(method=method, g_instruments=g_instruments)
    merge_data(method=method,
               g_instruments=g_instruments,
               horizon=1,
               category=1,
               fillna=False)
