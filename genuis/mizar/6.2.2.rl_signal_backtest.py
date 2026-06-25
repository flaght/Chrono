import json
from dotenv import load_dotenv
import multiprocessing as mp

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.bck001.engine import *

# signal_functions1 = {
#     "rollrank2_signal": {
#         "1001": {
#             "roll_num": 40,
#             "threshold": 0.90,
#         },
#         "1002": {
#             "roll_num": 50,
#             "threshold": 0.90,
#         },
#         "1003": {
#             "roll_num": 50,
#             "threshold": 0.95,
#         }
#     },
#     "rollrank1_signal": {
#         "1001": {
#             "roll_num": 30,
#             "threshold": 0.85,
#         },
#         "1002": {
#             "roll_num": 40,
#             "threshold": 0.90,
#         },
#         "1003": {
#             "roll_num": 50,
#             "threshold": 0.90,
#         },
#     }
# }
# signal_functions = {
#     "erband_signal": {
#         # 已验证基准组
#         "1001": {
#             "roll_num": 0,
#             "threshold": 0.30,
#             "upper": 0.85,
#         },
#         "1002": {
#             "roll_num": 0,
#             "threshold": 0.35,
#             "upper": 0.85,
#         },
#         "1003": {
#             "roll_num": 0,
#             "threshold": 0.30,
#             "upper": 0.90,
#         },

#         # 围绕 1002 的局部细化
#         "1004": {
#             "roll_num": 0,
#             "threshold": 0.32,
#             "upper": 0.85,
#         },
#         "1005": {
#             "roll_num": 0,
#             "threshold": 0.34,
#             "upper": 0.85,
#         },
#         "1006": {
#             "roll_num": 0,
#             "threshold": 0.36,
#             "upper": 0.85,
#         },
#         "1007": {
#             "roll_num": 0,
#             "threshold": 0.38,
#             "upper": 0.85,
#         },

#         # 检查 upper 是否需要更保守
#         "1008": {
#             "roll_num": 0,
#             "threshold": 0.35,
#             "upper": 0.80,
#         },
#         "1009": {
#             "roll_num": 0,
#             "threshold": 0.35,
#             "upper": 0.90,
#         },
#         "1010": {
#             "roll_num": 0,
#             "threshold": 0.35,
#             "upper": 0.95,
#         },

#         # 稍严格过滤，观察是否降低换手和回撤
#         "1011": {
#             "roll_num": 0,
#             "threshold": 0.40,
#             "upper": 0.85,
#         },
#         "1012": {
#             "roll_num": 0,
#             "threshold": 0.40,
#             "upper": 0.90,
#         },

#         # 稍宽松过滤，观察是否提升收益但增加噪声
#         "1013": {
#             "roll_num": 0,
#             "threshold": 0.28,
#             "upper": 0.85,
#         },
#         "1014": {
#             "roll_num": 0,
#             "threshold": 0.25,
#             "upper": 0.85,
#         },
#     }
# }

signal_functions = {
    "erband_signal": {
        "1002": {
            "roll_num": 0,
            "threshold": 0.35,
            "upper": 0.85,
        }
    }
}

def backtest_signal1(method, instruments, task_id, period, composite_method,
                     composite_id):
    trading_sessions = (("21:00", "23:00"), ("09:00", "10:15"),
                        ("10:30", "11:30"), ("13:30", "15:00"))
    # params = {
    #     "base_position": 100000,
    #     "lot_per_signal": 1,
    #     "entry_resampling_win": None,
    #     "max_daily_open_lots": None,
    #     "max_daily_open_lots_per_direction": None,
    #     "max_active_open_lots": None,
    #     "max_active_open_lots_per_direction": None,
    #     "min_abs_value": None,
    #     "block_same_direction_reentry": False,
    #     "block_opposite_direction_reentry": False,
    #     "extend_same_direction": False,
    # }
    
    params = {
        "lot_per_signal": 1,
        "entry_resampling_win":1,
        "max_active_lots":None
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
    val_data, test_data = load_er_data2(method=method,
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
