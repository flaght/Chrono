import os, sys, datetime, pdb
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

from lumina.evolution.fusion.actuator import Actuator
from kdutil.mongodb import MongoDBManager
from chaosmind.timing.sirius0001.workflow import WorkFlow
from toolix.params import load_sirius_params

from plugin.sirius.engine import Engine
from toolix.macro.contract import *


def create_time(begin_time, end_time, freq='1T'):
    return pd.date_range(start=begin_time, end=end_time, freq=freq)


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


def main1():
    pdb.set_trace()
    code = 'RB'
    symbol = 'rb2610'
    mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
    begin_time = pd.to_datetime('2026-05-06 14:55:00')
    end_time = pd.to_datetime('2026-05-23 10:35:00')
    trade_times = create_time(begin_time, end_time, freq='1T')
    actuator = Actuator(k_split=1)
    factors_infos = [
        "MADecay(30,'oi039_1_2_1')",
        "ADDED(EMA(15,'oi004_5_10_1'),MADiff(240,'oi004_1_2_1'))",
        "MA(10,'iv012_1_2_1')", "MDIFF(60,'iv012_2_3_1')",
        "MADiff(120,MOD('oi039_2_3_1','tc012_5_5_10_1'))"
    ]

    method = 'roll_zscore'
    win = 15

    for trade_time in trade_times:
        factors_data = fetch_factors(mongo_client=mongo_client,
                                     symbol=symbol,
                                     trade_time=trade_time,
                                     pos=120)

        # factors_data['oi039_1_2_1'] = factors_data['oi031_5_10_1']
        # factors_data['oi004_1_2_1'] = factors_data['oi004_5_10_1']
        # factors_data['oi039_2_3_1'] = factors_data['oi039_5_10_1']
        # factors_data['iv012_1_2_1'] = factors_data['iv012_5_10_1']
        # factors_data['iv012_2_3_1'] = factors_data['iv012_10_15_1']
        pdb.set_trace()
        original_factors, normal_factors = actuator.calculate(
            factors_infos=factors_infos,
            total_data=factors_data,
            method=method,
            win=win)


def main2():
    code = 'RB'
    symbol = 'rb2610'
    task_id = "1029921127239410"

    factors_infos, params = load_sirius_params(code=code, task_id=task_id)
    mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
    begin_time = pd.to_datetime('2026-05-06 14:55:00')
    end_time = pd.to_datetime('2026-05-23 10:35:00')
    trade_times = create_time(begin_time, end_time, freq='1T')
    pdb.set_trace()
    workflow = WorkFlow(directory=params['model_path'],
                        code=code,
                        symbol=symbol,
                        factors_infos=factors_infos,
                        softmax_temperature=params['softmax_temperature'],
                        min_open_signal_abs=params['min_open_signal_abs'],
                        method=params['method'],
                        win=params['win'])
    for trade_time in trade_times:
        total_data = fetch_factors(mongo_client=mongo_client,
                                   symbol=symbol,
                                   trade_time=trade_time,
                                   pos=120)
        if trade_time not in total_data['trade_time'].values:
            continue
        original_factors, normal_factors = workflow.create_factors(
            total_data=total_data)
        net_er_out = workflow.create_signals(trade_time=trade_time, data=normal_factors)
        print(trade_time, net_er_out)

def main3():
    begin_time = pd.to_datetime('2026-05-08 14:19:00')
    end_time = pd.to_datetime('2026-05-23 10:35:00')
    engine = Engine(codes=['RB'])
    trade_times = create_time(begin_time, end_time, freq='1T')
    for trade_time in trade_times:
        print(trade_time)
        engine.run(symbol='rb2610', trade_time=trade_time)
    
main3()
