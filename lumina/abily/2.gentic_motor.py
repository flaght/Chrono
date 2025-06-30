### 通过寻优算法 挖掘策略
import datetime, pdb, os, sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

#os.environ['INSTRUMENTS'] = 'ims'
#g_instruments = os.environ['INSTRUMENTS']

#sys.path.insert(0, os.path.abspath('../../'))

from kdutils.macro2 import *
from kdutils.operators_sets import operators_sets
from kdutils.callback import callback_fitness, callback_models
from lumina.genetic import Motor


def train(method, g_instruments):
    filename = os.path.join(base_path, method, g_instruments, 'merge',
                            "train_data.feather")
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    rootid = INDEX_MAPPING[INSTRUMENTS_CODES[g_instruments]]
    pdb.set_trace()
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    factors_data = factors_data.set_index('trade_time')
    factor_columns = [
        col for col in factors_data.columns if col not in [
            'trade_time', 'code', 'close', 'high', 'low', 'open', 'value',
            'volume', 'openint', 'vwap'
        ]
    ]

    population_size = 50
    tournament_size = 20
    strategy_settings = {
        'capital': 10000000,
        'commission': COST_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'slippage': SLIPPAGE_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[g_instruments]]
    }
    configure = {
        'n_jobs': 1,
        'population_size': population_size,
        'tournament_size': tournament_size,
        'init_depth': 8,
        'rootid': rootid,
        'generations': 10,
        'custom_params': {
            'g_instruments': g_instruments,
            'dethod': method,
            'tournament_size': tournament_size,
            'standard_score': 2,
            'strategy_settings': strategy_settings,
            'task_id': rootid,
            'method': PERFORMANCE_MAPPING[str(rootid)],
            'filter_custom': {
                'returns':
                FILTER_YEAR_MAPPING[INSTRUMENTS_CODES[g_instruments]]
            }
        }
    }
    motor = Motor(factor_columns=factor_columns,
                  callback_fitness=callback_fitness,
                  callback_save_model=callback_models)
    pdb.set_trace()
    motor.calculate(total_data=factors_data,
                    configure=configure,
                    operators_sets=operators_sets,
                    signals_sets=None,
                    strategies_sets=None)


if __name__ == '__main__':
    method = 'aicso2'
    g_instruments = 'ims'
    train(method=method, g_instruments=g_instruments)
