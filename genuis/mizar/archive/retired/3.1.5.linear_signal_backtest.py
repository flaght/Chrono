import os, json, pdb, copy
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from dotenv import load_dotenv
import multiprocessing as mp

load_dotenv()

from lib.uvx import *
from lib.cux001 import FactorEvaluate1
from lib.rl012.sandbox import PositionBacktester
from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.attr001.ftd001 import *
#from lib.bck002 import *
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

# _PARALLEL_MARKET_DATA = None

# def parallel_backtest(signal_data, name, parts, period, contract_multiplier,
#                       params, basic_path):
#     print(name)
#     signal_data = signal_data.rename(columns={'transformed': 'value'})
#     market_data = _PARALLEL_MARKET_DATA
#     # position_data = build_capped_locked_signals(
#     #     model_output=signal_data,
#     #     base_position=params['base_position'],
#     #     lot_per_signal=params['lot_per_signal'],
#     #     cooldown_bars=0,
#     #     hold_bars=period,
#     #     entry_resampling_win=params['entry_resampling_win'],
#     #     date_col='trade_date' if 'trade_date' in signal_data.columns else None,
#     #     max_daily_open_lots=params['max_daily_open_lots'],  #10,
#     #     max_daily_open_lots_per_direction=params[
#     #         'max_daily_open_lots_per_direction'],  #5,
#     #     max_active_open_lots=params['max_active_open_lots'],  #2,
#     #     max_active_open_lots_per_direction=params[
#     #         'max_active_open_lots_per_direction'],  #1,
#     #     extend_same_direction=params['extend_same_direction'],
#     #     min_abs_value=params['min_abs_value'],
#     #     block_same_direction_reentry=params['block_same_direction_reentry'],
#     #     block_opposite_direction_reentry=params[
#     #         'block_opposite_direction_reentry'])
#     pdb.set_trace()
#     position_data = build_paired_position_signals(
#         model_output=signal_data,
#         hold_bars=5,
#         lot_per_signal=1,
#         entry_resampling_win=5,
#         max_active_lots=None,
#         value_col="value",
#         signal_col="signal",
#         date_col="trade_date" if "trade_date" in signal_data.columns else None,
#         allow_overnight=True,
#     )

#     pb = PositionBacktester(market_data=market_data,
#                             contract_multiplier=contract_multiplier,
#                             slippage=0.001)
#     trade_records, daily_stats = pb.run(position_df=position_data, code='RB')
#     dirs1 = os.path.join(basic_path, parts[-3], parts[-2], name)
#     os.makedirs(dirs1, exist_ok=True)
#     pdb.set_trace()
#     position_data.to_feather(os.path.join(dirs1, "position_data.feather"))
#     trade_records.to_feather(os.path.join(dirs1, "trade_records.feather"))
#     daily_stats.to_feather(os.path.join(dirs1, "daily_stats.feather"))


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
    # base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
    #                           str(task_id), str(period), 'rl')
    # dirs1 = os.path.join(base_path1, "signal", "equal_weight", str(corr))

    # min_time = None
    # max_time = None
    # res = []
    # file_path = Path(dirs1)
    # for feat_file in file_path.rglob('*.feather'):
    #     print(feat_file)
    #     signal_data = pd.read_feather(feat_file)
    #     name = feat_file.parts[-1].split('.')[0]
    #     if 'test' not in name:
    #         continue
    #     parts = feat_file.parts
    #     min_time = signal_data['trade_time'].min(
    #     ) if min_time is None else min(signal_data['trade_time'].min(),
    #                                    min_time)
    #     max_time = signal_data['trade_time'].max(
    #     ) if max_time is None else max(signal_data['trade_time'].max(),
    #                                    max_time)
    #     res.append((name, parts, signal_data))

    # market_data = load_market_data(instruments=instruments,
    #                                begin_time=min_time,
    #                                end_time=max_time,
    #                                trading_sessions=trading_sessions)
    # global _PARALLEL_MARKET_DATA
    # _PARALLEL_MARKET_DATA = market_data
    # pdb.set_trace()

    # params = {
    #     'base_position': 10,
    #     'lot_per_signal': 1,
    #     'entry_resampling_win': period,
    #     'max_daily_open_lots': 6,  #每个交易日总共最多接受多少手“开暴露”信号
    #     'max_daily_open_lots_per_direction': 3,  # 每个交易日每个方向最多接受多少手
    #     'max_active_open_lots': 1,  #  同一时刻最多允许多少手暴露尚未恢复
    #     'max_active_open_lots_per_direction': 1,  # 同一时刻每个方向最多允许多少手暴露尚未恢复
    #     'min_abs_value': None,
    #     'block_same_direction_reentry': True,  # 同方向暴露未恢复前，不再接受同方向新信号。
    #     'block_opposite_direction_reentry': True,  # 任一反方向暴露未恢复前，不接受当前方向新信号。
    #     'extend_same_direction': True  # 若暴露到期 bar 又出现同方向信号，
    # }

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

    # basic_path = os.path.join(base_path, method, instruments, 'temp', 'model',
    #                           str(task_id), str(period), 'rl', 'backtest',
    #                           'equal_weight', Params.create_tag(params))

    # os.makedirs(basic_path, exist_ok=True)
    # config_path = os.path.join(basic_path, "config.json")
    # with open(config_path, "w", encoding="utf-8") as f:
    #     json.dump(params, f, indent=2, ensure_ascii=False)

    # args_for_backtest = [(item[2], item[0], item[1], period, 10, params,
    #                       basic_path) for item in res]

    # if len(res) <= 1:
    #     res = [
    #         parallel_backtest(signal_data=args_for_backtest[0][0],
    #                           name=args_for_backtest[0][1],
    #                           parts=args_for_backtest[0][2],
    #                           period=args_for_backtest[0][3],
    #                           contract_multiplier=args_for_backtest[0][4],
    #                           params=args_for_backtest[0][5],
    #                           basic_path=args_for_backtest[0][6])
    #     ]
    # else:
    #     try:
    #         ctx = mp.get_context("fork")
    #     except ValueError:
    #         ctx = mp.get_context()
    #     with ctx.Pool(processes=4) as pool:
    #         res = pool.starmap(parallel_backtest, args_for_backtest)
    # print(Params.create_tag(params))
    # parallel_backtest(signal_data=res[11][2],
    #                   name=res[11][0],
    #                   expression=res[11][1],
    #                   code='RB',
    #                   period=period,
    #                   max_position=1,
    #                   lot_per_signal=1,
    #                   contract_multiplier=10)


def metrics_signal1(method, instruments, task_id, period, composite_method,
                    composite_id):
    metrics_composite_signal(method=method,
                             instruments=instruments,
                             task_id=task_id,
                             period=period,
                             composite_method=composite_method,
                             composite_id=composite_id)
    # base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
    #                           str(task_id), str(period), 'rl')
    # dirs1 = os.path.join(base_path1, "signal", "equal_weight", str(corr))

    # file_path = Path(dirs1)
    # for feat_file in file_path.rglob('*.feather'):
    #     signal_data = pd.read_feather(feat_file)
    #     name = feat_file.parts[-1].split('.')[0]
    #     evaluate = FactorEvaluate1(factor_data=signal_data,
    #                                factor_name='signal',
    #                                ret_name='nxt1_ret_{0}h'.format(period),
    #                                roll_win=15,
    #                                fee=0.0,
    #                                scale_method='raw',
    #                                expression="{0}_{1}_{2}_{3}".format(
    #                                    feat_file.parts[-4],
    #                                    feat_file.parts[-3],
    #                                    feat_file.parts[-2], name),
    #                                resampling_win=period,
    #                                name=name)
    #     output = os.path.join(feat_file.parent, "metrics")
    #     os.makedirs(output, exist_ok=True)
    #     _ = evaluate.run()
    #     evaluate.plot_results()
    #     evaluate.save_results(output)


def create_signal1(method, instruments, task_id, period, composite_method,
                   composite_id):
    val_data, test_data = load_er_data(method=method,
                                       instruments=instruments,
                                       task_id=task_id,
                                       period=period,
                                       composite_method=composite_method,
                                       composite_id=composite_id,
                                       val_name='wf_val_data' if composite_method  == 'rl' else 'val_data',
                                       test_name='wf_test_data' if composite_method  == 'rl' else 'test_data')
    create_composite_signal(method=method,
                            instruments=instruments,
                            task_id=task_id,
                            period=period,
                            composite_method=composite_method,
                            composite_id=composite_id,
                            signal_functions=signal_functions,
                            val_data=val_data,
                            test_data=test_data)

    # def signal_to_save(data, name):
    #     signal_dt = create_signal(data=data.copy(),
    #                               signal_method=key1,
    #                               signal_params=params)
    #     total_data = signal_dt.merge(data, on=['trade_time', 'code'])
    #     filename = os.path.join(base_dirs,
    #                             "{0}_{1}.feather".format(key2, name))
    #     total_data.to_feather(filename)

    # base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
    #                           str(task_id), str(period), 'rl')

    # dirs_path = os.path.join(base_path1, "composite", "linear", 'equal_weight',
    #                          str(corr), 'data')
    # val_data = pd.read_feather(os.path.join(dirs_path, "val_data.feather"))
    # test_data = pd.read_feather(os.path.join(dirs_path, "test_data.feather"))
    # dirs1 = os.path.join(base_path1, "signal", "equal_weight")
    # for key1, functions in signal_functions.items():
    #     for key2, params in functions.items():
    #         print(key1, key2)
    #         base_dirs = os.path.join(dirs1, str(corr), key1)
    #         os.makedirs(base_dirs, exist_ok=True)
    #         signal_to_save(data=test_data,
    #                        name='test',
    #                        key1=key1,
    #                        key2=key2,
    #                        params=params)
    #         signal_to_save(data=val_data,
    #                        name='val',
    #                        key1=key1,
    #                        key2=key2,
    #                        params=params)

    # signal_method = 'quantile_signal'
    # signal_params = signal_functions[signal_method]["1001"]
    # pos_data = eval(signal_method)(factor_data=test_data.set_index(
    #     ['trade_time', 'code'])[['transformed']],
    #                                **signal_params)
    # pos_data = pos_data.stack()
    # pos_data.name = 'signal'
    # signals_df = pos_data.reset_index()

    # signals_df = signals_df.merge(
    #     test_data, on=['trade_time',
    #                    'code']).rename(columns={'transformed': 'value'})
    # position_data = build_position_signals(model_output=signals_df,
    #                                        hold_bars=5,
    #                                        max_position=1,
    #                                        lot_per_signal=1,
    #                                        cooldown_bars=5)

    # trading_sessions = (("21:00", "23:00"), ("09:00", "10:15"),
    #                     ("10:30", "11:30"), ("13:30", "15:00"))

    # market_data = load_market_data(instruments=instruments,
    #                                begin_time='2024-04-18',
    #                                end_time='2026-04-30',
    #                                trading_sessions=trading_sessions)
    # pdb.set_trace()
    # pb = PositionBacktester(market_data=market_data,
    #                         contract_multiplier=10,
    #                         slippage=0.1)
    # trade_records, daily_stats = pb.run(position_df=position_data, code='RB')
    # print('-->')


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
