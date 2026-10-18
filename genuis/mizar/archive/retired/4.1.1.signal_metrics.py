import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


from lumina.genetic.signal.method import *
from lumina.genetic.strategy.method import *

from kdutils.macro2 import *
from kdutils.tactix import Tactix

from lib.cux002 import StrategyEvaluate1
from lib.svx001 import create_position, read_factors, scale_factors

signal_functions = {
    'quantile_signal': {
        "1001": {
            'roll_num': 20,
            'threshold': 0.7
        },
        "1002": {
            'roll_num': 40,
            'threshold': 0.7
        }
    },
    'adaptive_signal': {
        "1001": {
            'roll_num': 25,
            'threshold': 0.9
        }
    }
}


def run(method, instruments, task_id, period, name, signal_method,
        signal_params):
    predict_data = read_factors(method=method,
                                instruments=instruments,
                                task_id=task_id,
                                period=period,
                                name=name)
    scale_factors(predict_data,
                  method='roll_zscore',
                  win=15,
                  factor_name='predict')

    strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]],
        'slippage': 0,
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }

    signal_method = signal_method
    signal_params = signal_functions[signal_method][signal_params]
    strategy_method = None
    strategy_params = None

    pos_data, total_data2 = create_position(predict_data=predict_data,
                                            signal_method=signal_method,
                                            signal_params=signal_params,
                                            strategy_method=strategy_method,
                                            strategy_params=strategy_params)

    eval1 = StrategyEvaluate1(
        pos_data=pos_data,
        total_data=total_data2,
        strategy_settings=strategy_settings,
        strategy_name=signal_method,
        ret_name='nxt1_ret_{0}h'.format(period),
    )

    eval1.run()


if __name__ == '__main__':
    variant = Tactix().start()
    run(method=variant.method,
        instruments=variant.instruments,
        task_id=variant.task_id,
        period=variant.period,
        name=variant.name,
        signal_method=variant.signal_method,
        signal_params=variant.signal_params)
