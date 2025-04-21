import pandas as pd
import sqlalchemy as sa
import os, pdb, sys, json
from dotenv import load_dotenv

load_dotenv()
os.environ['INSTRUMENTS'] = 'ifs'
g_instruments = os.environ['INSTRUMENTS']

sys.path.insert(0, os.path.abspath('../../'))

from kdutils.macro import *

from lumina.genetic import Actuator, StrategyTuple


def fetch_strategy(task_id):
    sql = """
        select name, formual, signal_method, signal_params, strategy_method, strategy_params,fitness from genetic_strategy where task_id={0} order by fitness desc limit 80
    """.format(task_id)
    engine = sa.create_engine(os.environ['DB_URL'])
    dt = pd.read_sql(sql=sql, con=engine)
    dt = dt[dt['fitness'] > 0]
    dt.to_dict(orient='records')
    strategies_data = [
        StrategyTuple(name=row.name,
                      formual=row.formual,
                      signal_method=row.signal_method,
                      signal_params=row.signal_params,
                      strategy_method=row.strategy_method,
                      strategy_params=row.strategy_params,
                      fitness=row.fitness) for row in dt.itertuples()
    ]
    return strategies_data


task_id = '20250414'
method = 'aicso1'
strategies_data = fetch_strategy(task_id)
pdb.set_trace()
actuator = Actuator(k_split=4)
filename = os.path.join(base_path, method, g_instruments, 'merge',
                        "val_data.feather")
factors_data = pd.read_feather(filename).sort_values(by=['trade_time', 'code'])
factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])

actuator.calculate(strategies_infos=strategies_data[:16], total_data=factors_data)
