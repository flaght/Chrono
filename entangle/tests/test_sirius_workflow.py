import os, sys, datetime, pdb
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

from kdutil.mongodb import MongoDBManager
from toolix.params import load_sirius_params
from toolix.macro.contract import *
from chaosmind.timing.sirius0003.workflow import WorkFlow

from plugin.sirius1.engine import Engine


def create_time(begin_time, end_time, freq='1T'):
    return pd.date_range(start=begin_time, end=end_time, freq=freq)


## 提取原始因子
def fetch_factors(mongo_client, symbol, trade_time, pos):
    rt = mongo_client['neutron'][RAW_FACTORS_TABLE].find(
        {
            'symbol': {
                "$in": [symbol]
            },
            "datetime": {
                "$lte": trade_time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }, {
            'datetime': 1,
            'symbol': 1,
            'name': 1,
            'value': 1
        }).sort([("datetime", -1)]).limit(pos * 1000)
    data = pd.DataFrame(rt)

    pivoted_df = data.pivot(index=['datetime', 'symbol'],
                            columns='name',
                            values='value').reset_index()
    pivoted_df['datetime'] = pd.to_datetime(pivoted_df['datetime'])
    return pivoted_df.rename(columns={
        'datetime': 'trade_time',
        'symbol': 'code'
    })


### 原始调用
def main1():
    begin_time = pd.to_datetime('2026-05-08 14:19:00')
    end_time = pd.to_datetime('2026-05-23 10:35:00')
    code = 'RB'
    symbol = 'rb2610'
    model_id = '1018806311332385'

    factors_infos, params = load_sirius_params(code=code,
                                               task_id=str(model_id))

    workflow = WorkFlow(directory=params['model_path'],
                        code=code,
                        symbol=symbol,
                        task_id=str(model_id),
                        factors_infos=factors_infos,
                        softmax_temperature=params['softmax_temperature'],
                        min_open_signal_abs=params['min_open_signal_abs'],
                        method=params['method'],
                        win=params['win'],
                        period=params['horizon'],
                        signal_method=params['signal_method'],
                        signal_params=params['signal_params'])

    workflow.initialization()

    mongo_client = MongoDBManager(uri=os.environ['MG_URI'])

    begin_time = pd.to_datetime('2026-05-06 14:55:00')
    end_time = pd.to_datetime('2026-05-23 10:35:00')
    trade_times = create_time(begin_time, end_time, freq='1T')

    pdb.set_trace()
    for trade_time in trade_times:
        total_data = fetch_factors(mongo_client=mongo_client,
                                   symbol=symbol,
                                   trade_time=trade_time,
                                   pos=120)
        if trade_time not in total_data['trade_time'].values:
            continue

        ### 历史修改
        total_data.rename(columns={
            'tc004_2_2_3_0': 'tc004_1_1_2_0',
            'tv004_10_15_0': 'tv004_1_2_0',
            'tv007_5_10_1': 'tv007_1_2_1'
        },
                          inplace=True)

        original_factors, normal_factors = workflow.create_factors(
            total_data=total_data)
        rt = workflow.create_values(trade_time=trade_time, data=normal_factors)
        events = workflow.conversion_signals(trade_time=trade_time,
                                             raw_action=pd.DataFrame([rt]),
                                             name='net_er_out')
        print('-->')


def main2():
    begin_time = pd.to_datetime('2026-06-29 21:00:00')
    end_time = pd.to_datetime('2026-06-30 15:00:00')
    trade_times = create_time(begin_time, end_time, freq='1T')
    engine = Engine(codes=['RB'])
    pdb.set_trace()
    for trade_time in trade_times:
        engine.run(symbol='rb2610',
                   trade_time=trade_time)


main2()
