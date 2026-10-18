#### 用于处理交易环境数据
#### 优先将交易环境

import datetime, pdb, re, os, time
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from dotenv import load_dotenv

load_dotenv()

from kdutils.data import *
from alphacopilot.calendars.api import *


def add_trade_date_column(market_data: pd.DataFrame,
                          time_name='datetime') -> pd.DataFrame:
    """
    根据 datetime 列，为 DataFrame 添加 trade_date 列。
    """
    # 确保 datetime 列是 Pandas 的 datetime 类型
    market_data.rename(columns={time_name: 'datetime'}, inplace=True)
    if not pd.api.types.is_datetime64_any_dtype(market_data['datetime']):
        dt_series = pd.to_datetime(market_data['datetime'])
    else:
        dt_series = market_data['datetime']

    # 获取自然日、小时、星期 (Monday=0, Sunday=6)
    dates = dt_series.dt.date
    hours = dt_series.dt.hour
    weekdays = dt_series.dt.weekday

    # 标记需要推延到下一个交易日的行：
    # 1. 小时 >= 18:00 (即 20:00 开始的夜盘)
    # 2. 或者在周末 (如周六凌晨 00:00 - 03:00)
    shift_mask = (hours >= 18) | (weekdays >= 5)

    # 提取所有需要计算“下一个交易日”的唯一自然日，避免重复计算
    unique_shift_dates = dates[shift_mask].unique()
    shift_mapping = {}

    for d in unique_shift_dates:
        # 计算下一个交易日
        next_td = advanceDateByCalendar('china.sse', d, '1b')
        # 处理返回结果可能是 datetime.datetime 的情况，统一转为 date
        if hasattr(next_td, 'date'):
            next_td = next_td.date()
        shift_mapping[d] = next_td

    # 初始化 trade_date 列为自然日
    market_data['trade_date'] = dates

    # 将需要推延的行，映射为计算好的下一个交易日
    if len(shift_mapping) > 0:
        market_data.loc[shift_mask,
                        'trade_date'] = dates[shift_mask].map(shift_mapping)
    market_data['trade_date'] = pd.to_datetime(market_data['trade_date'])
    return market_data


### 从交易环境获取
def fetch_trade(begin_date, end_date):
    market_data = fetch_trader_market0(
        begin_date=begin_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'))
    return market_data


### 从备份环境获取
def fetch_bench(begin_date, end_date, codes):
    return fetch_bench_market0(begin_date=begin_date,
                               end_date=end_date,
                               codes=codes,
                               forced_alignment=True)


### 从投研环境获取
def fetch_research(begin_date, end_date, codes):
    return fetch_local_market0(base_path=os.environ['BAR_FUT_DIRS'],
                               begin_date=begin_date,
                               end_date=end_date,
                               codes=codes,
                               is_trading=True)


def save_data(market_data):
    ### 合约 + 日期保存
    grouped = market_data.groupby(by=['trade_date', 'symbol'])
    for k, g in grouped:
        dirs = os.path.join(
            os.environ['TRADE_FUT_DIRS'],
            k[0] if isinstance(k[0], str) else k[0].strftime('%Y%m%d'))
        os.makedirs(dirs, exist_ok=True)
        filename = os.path.join(
            dirs, "{0}_{1}.feather".format(
                k[1],
                k[0] if isinstance(k[0], str) else k[0].strftime('%Y%m%d')))
        print(filename)
        g.drop(['trade_date'],
               axis=1).reset_index(drop=True).to_feather(filename)


### 从备份环境获取
def start3():
    end_date = advanceDateByCalendar('china.sse', datetime.datetime.now(),
                                     '1b')
    begin_date = advanceDateByCalendar('china.sse', end_date, '-26b')

    ### 多移一天，夜盘情况
    start_date = advanceDateByCalendar('china.sse', begin_date, '-2b')
    codes = ['RB', 'NI', 'V', 'M', 'VI', 'MA']
    market_data = fetch_bench(begin_date=start_date,
                              end_date=end_date,
                              codes=codes)
    market_data = add_trade_date_column(market_data=market_data,
                                        time_name='trade_time')
    pdb.set_trace()
    market_data = market_data[market_data['trade_date'].between(begin_date, end_date)]
    save_data(market_data)


### 从投研环境获取
def start2():
    begin_date = '2021-11-25'
    end_date = "2024-01-01"
    codes = ['RB', 'NI', 'V', 'M', 'VI', 'MA']
    market_data = fetch_research(begin_date=begin_date,
                                 end_date=end_date,
                                 codes=codes)
    pdb.set_trace()
    save_data(market_data)


### 从交易环境获取
def start1():
    end_date = advanceDateByCalendar('china.sse', datetime.datetime.now(),
                                     '1b')  # mongo db 是区间
    begin_date = advanceDateByCalendar('china.sse', datetime.datetime.now(),
                                       '-18b')
    market_data = fetch_trade(begin_date=begin_date, end_date=end_date)
    market_data1 = add_trade_date_column(market_data=market_data,
                                         time_name='datetime')
    pdb.set_trace()
    save_data(market_data1)


# def compare():
#     pdb.set_trace()
#     end_date = advanceDateByCalendar('china.sse', datetime.datetime.now(),
#                                      '-40b')  # mongo db 是区间
#     begin_date = advanceDateByCalendar('china.sse', end_date,
#                                        '-18b')
#     codes = ['RB', 'NI', 'V', 'M', 'VI', 'MA']
#     trade_market_data = fetch_trade(begin_date=begin_date, end_date=end_date)
#     trade_market_data = add_trade_date_column(market_data=trade_market_data)

#     research_data = fetch_research(begin_date=begin_date.strftime('%Y-%m-%d'),
#                                    end_date=end_date.strftime('%Y-%m-%d'),
#                                    codes=codes)
#     print('-->')
#     print('-->')

if __name__ == '__main__':
    start3()
