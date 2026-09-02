import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Iterable, List
from dotenv import load_dotenv

load_dotenv()

from lib.lqc001 import *
# from kdutils.data import *
# from ultron.tradingday import *
# from ultron.sentry.api import *

codes1 = [
    'RB', 'J', 'ZN', 'AU', 'AG', 'SC', 'RM', 'RU', 'FG', 'SA', 'I', 'JM', 'HC',
    'SF', 'SM', 'TA', 'MA', 'EG', 'L', 'PP', 'V', 'FU', 'BU', 'A', 'Y', 'M',
    'OI', 'P', 'C', 'CS', 'SR', 'CF', 'JD', 'NI', 'AL', 'CU', 'PB'
]

OUTPUT_COLUMNS = [
    "code",
    "score",
    "is_high_vol",
    "rows",
    "active_days",
    "start",
    "end",
    "avg_amp_pct",
    "p80_amp_pct",
    "atr_pct",
    "realized_vol_ann_pct",
    "roundtrip_cost_bp",
    "amp_to_cost",
    "opportunity_after_cost_pct",
    "trend_efficiency",
    "jump_share",
    "median_volume",
    "median_value",
    "median_openint",
]


def fetch_data(begin_time, end_time, adjusted_method):
    basic_infos = fetch_basic(codes=codes1,
                              begin_time=begin_time,
                              end_time=end_time)
    
    codes = basic_infos['code'].unique().tolist()
    market_data = fetch_market(codes=codes,
                               begin_time=begin_time,
                               end_time=end_time,
                               adjusted_method=adjusted_method)

    market_data = market_data.sort_values(by=['trade_time', 'code']).merge(
        basic_infos, on=['code', 'symbol'], how='left')

    ##过滤非标准数据
    dates = interval_trading_date(begin_date=market_data['trade_time'].min(),
                                  end_date=market_data['trade_time'].max())

    market_data = market_data.drop_duplicates(subset=['trade_time', 'code'])
    market_data = market_data[pd.to_datetime(
        market_data['trade_time']).dt.normalize().isin(dates)]
    return market_data, basic_infos


def start(trade_time,
          slippage_bp=0.001,
          window_days=60,
          min_periods=30,
          min_rows=60,
          min_amp_pct=0.1,
          min_atr_pct=0.1,
          min_amp_to_cost=1.0,
          high_vol_top_n=30,
          high_vol_top_pct=None,
          adjusted_method='pcr'):

    begin_time = advanceDateByCalendar('china.sse', trade_time, '-90b')
    #start_time = advanceDateByCalendar('china.sse', begin_time, '-1b')
    market_data, basic_infos = fetch_data(begin_time=begin_time,
                                          end_time=trade_time,
                                          adjusted_method=adjusted_method)

    market_matrix = market_data.set_index(['trade_time', 'code']).unstack()

    default_cost = basic_infos.groupby(
        'code')['tradeCommiNum'].median() + slippage_bp

    output_data = calc_indicator(market_matrix=market_matrix,
                                 min_rows=min_rows,
                                 slippage_bp=slippage_bp,
                                 default_cost=default_cost,
                                 window_days=window_days,
                                 min_periods=min_periods,
                                 min_amp_pct=min_amp_pct,
                                 min_atr_pct=min_atr_pct,
                                 min_amp_to_cost=min_amp_to_cost,
                                 high_vol_top_n=high_vol_top_n,
                                 high_vol_top_pct=high_vol_top_pct)
    pdb.set_trace()
    print('--。')


if __name__ == '__main__':
    start(trade_time=datetime.datetime(2026, 6, 12))
