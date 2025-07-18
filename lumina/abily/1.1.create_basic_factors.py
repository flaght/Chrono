### 创建lumina所有的因子
import datetime
import pdb, os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

#os.environ['INSTRUMENTS'] = 'ims'
#g_instruments = os.environ['INSTRUMENTS']

from alphacopilot.api.calendars import advanceDateByCalendar
from kdutils.ttimes import get_dates
from kdutils.macro import base_path
from config.contract import INSTRUMENTS_CODES
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


def callback_save(instruments, factors_data, name, method, start_date, end_date):
    cond1 = (factors_data.index.get_level_values(
        level=0) >= start_date) & (factors_data.index.get_level_values(
            level=0) <= (datetime.datetime.strptime(end_date, '%Y-%m-%d') +
                         datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    factors_data = factors_data[cond1]
    ff = factors_data.sort_index().reset_index()
    ff1 = ff  #ff.set_index(['trade_time', 'code']).unstack()
    dirs = os.path.join(base_path, method, instruments, 'factors')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs,
                            '{0}_factors.feather'.format(name.split('.')[-1]))
    print(filename)
    ff1.sort_index().reset_index(drop=True).to_feather(filename)


def calculate_factors(data, callback, instruments, method, start_date, end_date):

    def run(data, i00, callback, instruments, method, start_date, end_date):
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
                 instruments=instruments,
                 name=i00.__name__,
                 method=method,
                 start_date=start_date,
                 end_date=end_date)

    for i00 in [
            i001, i002, i003, i004, i005, i006, i007, i008, i009, i010, i011,
            i012, i013
    ]:
        run(data=data,
            i00=i00,
            callback=callback,
            instruments=instruments,
            method=method,
            start_date=start_date,
            end_date=end_date)


def main(method, instruments):
    start_date, end_date = get_dates(method)
    start_time = advanceDateByCalendar('china.sse', start_date,
                                       '-{0}b'.format(1)).strftime('%Y-%m-%d')
    data = fetch_main_market(begin_date=start_time,
                             end_date=end_date,
                             codes=[INSTRUMENTS_CODES[instruments]])
    data = data.set_index(['trade_time', 'code']).unstack()
    calculate_factors(data,
                      instruments=instruments,
                      callback=callback_save,
                      method=method,
                      start_date=start_date,
                      end_date=end_date)


def merge(method, instruments):
    pdb.set_trace()
    base_dirs = os.path.join(base_path, method, instruments, 'factors')
    res = []
    for root, dirs, files in os.walk(base_dirs):
        for file in files:
            print(file)
            if file.endswith('.feather') and file != 'factors_data.feather':
                factor_file = os.path.join(root, file)
                factor_data = pd.read_feather(factor_file)
                res.append(factor_data.set_index(['trade_time', 'code']))
    factors_data = pd.concat(res, axis=1).sort_index()
    factors_data = factors_data.unstack().fillna(method='ffill')
    factors_data = factors_data.stack().dropna().reset_index()
    start_date = factors_data['trade_time'].min().strftime('%Y-%m-%d %H:%M:%S')
    end_date = factors_data['trade_time'].max().strftime('%Y-%m-%d %H:%M:%S')
    data = fetch_main_market(begin_date=start_date,
                             end_date=end_date,
                             codes=[INSTRUMENTS_CODES[instruments]])
    pdb.set_trace()
    factors_data = factors_data.merge(data[[
        'trade_time', 'code', 'close', 'high', 'low', 'open', 'value',
        'volume', 'openint', 'vwap'
    ]],
                                      on=['trade_time', 'code'])
    factors_data['trade_time'] = pd.to_datetime(
        factors_data['trade_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    factors_data = factors_data.sort_values(by=['trade_time', 'code'])

    times = factors_data['trade_time'].unique().tolist()

    len1 = round(len(times) * 0.7)  # 70%部分
    len2 = round(len(times) * 0.2)  # 20%部分
    len3 = len(times) - len1 - len2

    ## 训练集
    pdb.set_trace()
    train_data = factors_data[factors_data['trade_time'].isin(times[:len1])]
    val_data = factors_data[factors_data['trade_time'].isin(times[len1:len1 +
                                                                  len2])]
    test_data = factors_data[factors_data['trade_time'].isin(times[len1 +
                                                                   len2:])]
    ## 校验集
    ## 测试集
    ### 切割数据
    target_dir = os.path.join(base_path, method, instruments, 'basic')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    train_data.reset_index(drop=True).to_feather(
        os.path.join(base_path, method, instruments, 'basic',
                     'train_data.feather'))
    val_data.reset_index(drop=True).to_feather(
        os.path.join(base_path, method, instruments, 'basic',
                     'val_data.feather'))
    test_data.reset_index(drop=True).to_feather(
        os.path.join(base_path, method, instruments, 'basic',
                     'test_data.feather'))


main(method='aicso0', instruments='rbb')
merge(method='aicso0', instruments='rbb')
