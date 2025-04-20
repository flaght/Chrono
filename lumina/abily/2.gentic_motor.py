import datetime, pdb, os, sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

os.environ['INSTRUMENTS'] = 'ims'
g_instruments = os.environ['INSTRUMENTS']

sys.path.insert(0, os.path.abspath('../../'))

from kdutils.macro import *
from kdutils.operators_sets import operators_sets
from kdutils.callback import callback_fitness, callback_models
from lumina.genetic.motor import Motor


def train(method):
    filename = os.path.join(base_path, method, g_instruments, 'merge',
                            "train_data.feather")
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    rootid = INDEX_MAPPING[instruments_codes[g_instruments][0]]
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
        'commission': COST_MAPPING[instruments_codes[g_instruments][0]],
        'slippage': SLIPPAGE_MAPPING[instruments_codes[g_instruments][0]],
        'size': CONT_MULTNUM_MAPPING[instruments_codes[g_instruments][0]]
    }
    pdb.set_trace()
    configure = {
        'n_jobs': 4,
        'population_size': population_size,
        'tournament_size': tournament_size,
        'init_depth': 4,
        'rootid': rootid,
        'generations':15,
        'custom_params': {
            'g_instruments': g_instruments,
            'dethod': method,
            'tournament_size': tournament_size,
            'standard_score': 2,
            'strategy_settings': strategy_settings
        }
    }
    motor = Motor(factor_columns=factor_columns,
                  callback_fitness=callback_fitness,
                  callback_save_model=callback_models)

    motor.calculate(total_data=factors_data,
                    configure=configure,
                    operators_sets=operators_sets,
                    signals_sets=None,
                    strategies_sets=None)


if __name__ == '__main__':
    # rootid = datetime.datetime.now().strftime("%Y%m%d")
    ### IF：20250415001
    method = 'aicso2'
    train(method)
