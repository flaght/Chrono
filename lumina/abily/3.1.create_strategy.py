import pandas as pd
import sqlalchemy as sa
import os, pdb, sys, json
from ultron.factor.genetic.geneticist.operators import *
from dotenv import load_dotenv

load_dotenv()
os.environ['INSTRUMENTS'] = 'ifs'
g_instruments = os.environ['INSTRUMENTS']

sys.path.insert(0, os.path.abspath('../../'))

from kdutils.macro import *
from lumina.genetic.signal.method import *
from lumina.genetic.strategy.method import *


def create_position(total_data, strategy):
    factors_data = calc_factor(expression=strategy.formual,
                               total_data=total_data.set_index(['trade_time']),
                               key='code',
                               indexs=[])
    factors_data1 = factors_data.reset_index().set_index(
        ['trade_time', 'code'])
    total_data1 = total_data.set_index(['trade_time', 'code']).unstack()
    pos_data = eval(strategy.signal_method)(factor_data=factors_data1,
                                            **json.loads(
                                                strategy.signal_params))
    pos_data1 = eval(strategy.strategy_method)(signal=pos_data,
                                               total_data=total_data1,
                                               **json.loads(
                                                   strategy.strategy_params))
    return pos_data1


def fetch_strategy(task_id):
    sql = """
        select formual, signal_method, signal_params, strategy_method, strategy_params from genetic_strategy where task_id={0} order by fitness desc limit 80
    """.format(task_id)
    engine = sa.create_engine(os.environ['DB_URL'])
    dt = pd.read_sql(sql=sql, con=engine)
    return dt


task_id = '20250414'
method = 'aicso1'
strategies_dt = fetch_strategy(task_id)
filename = os.path.join(base_path, method, g_instruments, 'merge',
                        "val_data.feather")
factors_data = pd.read_feather(filename).sort_values(by=['trade_time', 'code'])
factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
for row in strategies_dt.itertuples():
    create_position(total_data=factors_data, strategy=row)
