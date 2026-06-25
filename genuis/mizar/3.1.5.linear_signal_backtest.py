import os, json, pdb, copy
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from dotenv import load_dotenv
import multiprocessing as mp

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.bck001.engine import *

# signal_functions = {
#     "band_signal": {
#         "1001": {
#             "roll_num": 0,
#             "threshold": 0.05,
#         },
#         "1002": {
#             "roll_num": 0,
#             "threshold": 0.10,
#         },
#         "1003": {
#             "roll_num": 0,
#             "threshold": 0.15,
#         },
#         "1004": {
#             "roll_num": 0,
#             "threshold": 0.20,
#         },
#     },
#     "gate_signal": {
#         "1001": {
#             "roll_num": 20,
#             "threshold": 0.75,
#         },
#         "1002": {
#             "roll_num": 24,
#             "threshold": 0.80,
#         },
#         "1003": {
#             "roll_num": 30,
#             "threshold": 0.85,
#         },
#         "1004": {
#             "roll_num": 40,
#             "threshold": 0.90,
#         },
#     },
#     "rollrank1_signal": {
#         "1001": {
#             "roll_num": 20,
#             "threshold": 0.80,
#         },
#         "1002": {
#             "roll_num": 24,
#             "threshold": 0.85,
#         },
#         "1003": {
#             "roll_num": 30,
#             "threshold": 0.90,
#         },
#     },
#     "rollrank2_signal": {
#         "1001": {
#             "roll_num": 20,
#             "threshold": 0.75,
#         },
#         "1002": {
#             "roll_num": 24,
#             "threshold": 0.80,
#         },
#         "1003": {
#             "roll_num": 30,
#             "threshold": 0.85,
#         },
#     }
# }

signal_functions = {
    "rollrank2_signal": {
        "1002": {
            "roll_num": 24,
            "threshold": 0.80,
        }
    }
}


def backtest_signal1(method, instruments, task_id, period, composite_method,
                     composite_id):
    trading_sessions = (("21:00", "23:00"), ("09:00", "10:15"),
                        ("10:30", "11:30"), ("13:30", "15:00"))
    params = {
        "base_position": 100000,
        "lot_per_signal": 1,
        "entry_resampling_win": None,
        "max_daily_open_lots": None,
        "max_daily_open_lots_per_direction": None,
        "max_active_open_lots": None,
        "max_active_open_lots_per_direction": None,
        "min_abs_value": None,
        "block_same_direction_reentry": False,
        "block_opposite_direction_reentry": False,
        "extend_same_direction": False,
    }

    backtest_composite_signal(method=method,
                              instruments=instruments,
                              task_id=task_id,
                              period=period,
                              composite_method=composite_method,
                              composite_id=composite_id,
                              params=params,
                              trading_sessions=trading_sessions)


def metrics_signal1(method, instruments, task_id, period, composite_method,
                    composite_id):
    metrics_composite_signal(method=method,
                             instruments=instruments,
                             task_id=task_id,
                             period=period,
                             composite_method=composite_method,
                             composite_id=composite_id)

def create_signal1(method, instruments, task_id, period, composite_method,
                   composite_id):
    val_data, test_data = load_er_data1(method=method,
                                       instruments=instruments,
                                       task_id=task_id,
                                       period=period,
                                       composite_method=composite_method,
                                       composite_id=composite_id,
                                       val_name='val_data',
                                       test_name='test_data')
    create_composite_signal(method=method,
                            instruments=instruments,
                            task_id=task_id,
                            period=period,
                            composite_method=composite_method,
                            composite_id=composite_id,
                            signal_functions=signal_functions,
                            val_data=val_data,
                            test_data=test_data)


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == 'build':
        create_signal1(method=variant.method,
                       instruments=variant.instruments,
                       task_id=variant.task_id,
                       period=variant.period,
                       composite_id=variant.composite_id,
                       composite_method=variant.composite_method)
    elif variant.form == 'metrics':
        metrics_signal1(method=variant.method,
                        instruments=variant.instruments,
                        task_id=variant.task_id,
                        period=variant.period,
                        composite_id=variant.composite_id,
                        composite_method=variant.composite_method)
    elif variant.form == 'backtest':
        backtest_signal1(method=variant.method,
                         instruments=variant.instruments,
                         task_id=variant.task_id,
                         period=variant.period,
                         composite_id=variant.composite_id,
                         composite_method=variant.composite_method)