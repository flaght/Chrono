import pandas as pd
import sqlalchemy as sa
import os, pdb, sys, json

from dotenv import load_dotenv

load_dotenv()
os.environ['INSTRUMENTS'] = 'ims'
g_instruments = os.environ['INSTRUMENTS']

from lumina.genetic import Thruster
from lumina.genetic import StrategyTuple

from kdutils.macro import *


def fetch_data(method):
    filename = os.path.join(base_path, method, g_instruments, 'merge',
                            "val_data.feather")
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    return factors_data


def fetch_strategy(task_id, threshold=1.0):
    sql = """
        select name, formual, signal_method, signal_params, strategy_method, fitness, strategy_params from genetic_strategy where task_id={0} order by fitness desc limit 80
    """.format(task_id)
    engine = sa.create_engine(os.environ['DB_URL'])
    dt = pd.read_sql(sql=sql, con=engine)
    dt = dt[dt['fitness'] > threshold]
    dt = [StrategyTuple(**d1) for d1 in dt.to_dict(orient='records')]
    return dt


method = 'aicso2'
k_split = 4
task_id = INDEX_MAPPING[instruments_codes[g_instruments][0]]
strategies_dt = fetch_strategy(task_id)
val_data = fetch_data(method=method)
val_data['trade_time'] = pd.to_datetime(val_data['trade_time'])
strategy_settings = {
    'capital': 10000000,
    'commission': COST_MAPPING[instruments_codes[g_instruments][0]],
    'slippage': SLIPPAGE_MAPPING[instruments_codes[g_instruments][0]],
    'size': CONT_MULTNUM_MAPPING[instruments_codes[g_instruments][0]]
}

thruster = Thruster(k_split=1)

results = thruster.calculate(strategies_infos=strategies_dt,
                             strategy_setting=strategy_settings,
                             total_data=val_data)
dts = [{
    'name': r1.name,
    'annual_return': r1.annual_return,
    'annual_volatility': r1.annual_volatility,
    'calmar': r1.calmar,
    'sharpe': r1.sharpe,
    'max_drawdown': r1.max_drawdown,
    'sortino': r1.sortino
} for r1 in results]
pdb.set_trace()
print(dts)
