import os, sys, pdb, re, math, json
import sqlalchemy as sa
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath('../'))

from lumina.genetic import Thruster
from lumina.genetic import StrategyTuple

base_path = '/workspace/worker/pj/Chrono/lumina/abily/records'


def fetch_data(method):
    filename = os.path.join(base_path, method, 'ifs', 'merge',
                            "val_data.feather")
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    return factors_data


def fetch_strategy(task_id, threshold=0.6):
    sql = """
        select name, formual, signal_method, signal_params, strategy_method, fitness, strategy_params from genetic_strategy where task_id={0} order by fitness desc limit 80
    """.format(task_id)
    engine = sa.create_engine(
        'mysql+mysqlconnector://neutron:Jc2D6sip@172.17.0.1:3306/quant')
    dt = pd.read_sql(sql=sql, con=engine)
    dt = dt[dt['fitness'] > threshold][0:10]
    dt = [StrategyTuple(**d1) for d1 in dt.to_dict(orient='records')]
    return dt


factors_data = fetch_data('aicso1')
strategy = fetch_strategy(task_id=100001, threshold=0.6)

strategy_settings = {
    'capital': 10000000,
    'commission': 0.000023,
    'slippage': 0.0001,
    'size': 300
}

thruster = Thruster(k_split=1)

thruster.calculate(strategies_infos=strategy,
                   strategy_setting=strategy_settings,
                   total_data=factors_data)
