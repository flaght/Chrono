import os, json, pdb, copy
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from dotenv import load_dotenv
import multiprocessing as mp

load_dotenv()

from chaosmind.timing.sirius1001.workflow import WorkFlow
from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.bck001.engine import *
from lib.bck001.common import *

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


def _workflow(data, factors_infos, instruments, model_id, period, params, name,
              dirs1):
    pdb.set_trace()
    wf = WorkFlow(factors_infos=factors_infos,
                  code=INSTRUMENTS_CODES[instruments],
                  symbol="{0}9999".format(instruments.lower()),
                  task_id=model_id,
                  period=period,
                  signal_method=params['signal_method'],
                  signal_params=params['signal_params'],
                  method=params['method'],
                  win=params['win'])
    workflow_creator(wf=wf,
                     data=data,
                     params=params,
                     period=period,
                     name=name,
                     dirs1=dirs1)

    # wf.initialization()

    # res1 = []
    # res2 = []
    # i = 0
    # for i in range(0, data.shape[0]):
    #     if i <= params['signal_params']['roll_num']:
    #         continue
    #     raw_action = data.loc[i - params['signal_params']['roll_num']:i]
    #     trade_time = raw_action.loc[i]['trade_time']
    #     print(trade_time)
    #     signal, events = wf.wrapper(trade_time=trade_time,
    #                                 raw_action=raw_action,
    #                                 name='transformed')
    #     # pdb.set_trace()
    #     # if i > 2000:
    #     #      break
    #     res1.append(signal)
    #     res2.extend(events)

    # signal_data = pd.DataFrame(res1)
    # signal_data = signal_data.merge(data, on=['trade_time', 'code'])
    # filename = os.path.join(dirs1, "{0}.feather".format(name))
    # signal_data.to_feather(filename)

    # events_data = pd.DataFrame(res2)

    # ## 全保留，在绩效时候 再做相应处理
    # #events_data = events_data[events_data['signal_type'] != 'close']

    # events_data = events_data.merge(data, on=['trade_time', 'code'])
    # filename = os.path.join(dirs1, "event_{0}.feather".format(name))
    # events_data.to_feather(filename)


# def _backtest_event(instruments, period, name, base_dir, output_dir):
#     pdb.set_trace()
#     trading_sessions = (("21:00", "23:00"), ("09:00", "10:15"),
#                         ("10:30", "11:30"), ("13:30", "15:00"))

#     filename = os.path.join(os.path.join(base_dir, "data"),
#                             "event_{0}.feather".format(name))
#     event_data = pd.read_feather(filename)
#     pdb.set_trace()
#     min_time = event_data['trade_time'].min()
#     max_time = event_data['trade_time'].max()
#     market_data = load_market_data(instruments=instruments,
#                                    begin_time=min_time,
#                                    end_time=max_time,
#                                    trading_sessions=trading_sessions)

#     event_data = event_data[[
#         'trade_time', 'code', 'direction', 'numbers', 'signal_type'
#     ]].copy()

#     event_data = event_data[event_data['signal_type'].isin(['open',
#                                                             'close'])].copy()
#     event_data['trade_time'] = pd.to_datetime(event_data['trade_time'])
#     event_data['date'] = event_data['trade_time'].dt.normalize()
#     event_data['min_time'] = event_data['trade_time'].dt.strftime('%H%M')

#     pb = PositionBacktester(market_data=market_data,
#                             contract_multiplier=10,
#                             slippage=0.001)
#     trade_records, daily_stats = pb.run(position_df=event_data, code='RB')
#     dirs1 = os.path.join(output_dir, name)
#     os.makedirs(dirs1, exist_ok=True)
#     event_data.reset_index(drop=True).to_feather(
#         os.path.join(dirs1, "position_data.feather"))
#     trade_records.to_feather(os.path.join(dirs1, "trade_records.feather"))
#     daily_stats.to_feather(os.path.join(dirs1, "daily_stats.feather"))

# def _metrics_event(base_dir, period, name):
#     filename = os.path.join(os.path.join(base_dir, "data"),
#                             "event_{0}.feather".format(name))
#     event_data = pd.read_feather(filename)
#     output = os.path.join(base_dir, "metrics")
#     evaluate = FactorEvaluate1(
#         factor_data=event_data,
#         factor_name='position_direction',
#         ret_name='nxt1_ret_{0}h'.format(period),
#         roll_win=15,
#         fee=0.00001,
#         scale_method='raw',
#         expression="final_{0}".format(name),
#         resampling_win=1,  #period, 事件产生的开仓信号到平仓，持仓5分钟，当前可以用于连续开仓。模拟回测效果
#         name="event_{0}".format(name))
#     os.makedirs(output, exist_ok=True)
#     _ = evaluate.run()
#     evaluate.plot_results()
#     evaluate.save_results(output)

# def _metrics_signal(base_dir, period, name):
#     filename = os.path.join(os.path.join(base_dir, "data"),
#                             "{0}.feather".format(name))
#     signal_data = pd.read_feather(filename)
#     output = os.path.join(base_dir, "metrics")
#     evaluate = FactorEvaluate1(factor_data=signal_data,
#                                factor_name='transformed',
#                                ret_name='nxt1_ret_{0}h'.format(period),
#                                roll_win=15,
#                                fee=0.0,
#                                scale_method='raw',
#                                expression="final_{0}".format(name),
#                                resampling_win=period,
#                                name=name)
#     os.makedirs(output, exist_ok=True)
#     _ = evaluate.run()
#     evaluate.plot_results()
#     evaluate.save_results(output)


def auto_workflow(method, instruments, task_id, period, composite_method,
                  composite_id, model_id):

    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dirs1 = os.path.join(base_path1, "signal", "final", composite_method,
                         model_id, "data")
    pdb.set_trace()
    os.makedirs(dirs1, exist_ok=True)
    factors_infos, params = load_sirius_params(
        code=INSTRUMENTS_CODES[instruments], task_id=str(model_id))

    ### WorkFlow 对应 具体的方法，比如当前WorkFlow 对应等权

    ### 加载 wf er
    val_data, test_data = load_er_data1(method=method,
                                        instruments=instruments,
                                        task_id=task_id,
                                        period=period,
                                        composite_method=composite_method,
                                        composite_id=composite_id,
                                        val_name='val_data',
                                        test_name='test_data',
                                        category='wf')

    # _workflow(data=val_data,
    #           factors_infos=factors_infos,
    #           instruments=instruments,
    #           model_id=model_id,
    #           period=period,
    #           params=params,
    #           name='val',
    #           dirs1=dirs1)

    _workflow(data=test_data,
              factors_infos=factors_infos,
              instruments=instruments,
              model_id=model_id,
              period=period,
              params=params,
              name='test',
              dirs1=dirs1)


def metrics_workflow(method, instruments, task_id, period, composite_method,
                     composite_id, model_id):

    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dirs1 = os.path.join(base_path1, "signal", "final", composite_method,
                         model_id)

    metrics_event(base_dir=dirs1,
                  period=period,
                  name='val',
                  roll_win=15,
                  fee=0.00001)
    metrics_event(base_dir=dirs1,
                  period=period,
                  name='test',
                  roll_win=15,
                  fee=0.00001)

    metrics_signal(base_dir=dirs1,
                   period=period,
                   name='val',
                   roll_win=15,
                   fee=0.0)
    metrics_signal(base_dir=dirs1,
                   period=period,
                   name='test',
                   roll_win=15,
                   fee=0.0)


def backtest_workflow(method, instruments, task_id, period, composite_method,
                      composite_id, model_id):
    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dirs1 = os.path.join(base_path1, "signal", "final", composite_method,
                         model_id)
    dirs2 = os.path.join(base_path1, "backtest", "final", composite_method,
                         model_id)
    backtest_event(instruments=instruments,
                   period=period,
                   name='val',
                   base_dir=dirs1,
                   output_dir=dirs2)

    backtest_event(instruments=instruments,
                   period=period,
                   name='test',
                   base_dir=dirs1,
                   output_dir=dirs2)


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
    if variant.form == 'build':  # or
        create_signal1(method=variant.method,
                       instruments=variant.instruments,
                       task_id=variant.task_id,
                       period=variant.period,
                       composite_id=variant.composite_id,
                       composite_method=variant.composite_method)
    elif variant.form == 'metrics':  # or
        metrics_signal1(method=variant.method,
                        instruments=variant.instruments,
                        task_id=variant.task_id,
                        period=variant.period,
                        composite_id=variant.composite_id,
                        composite_method=variant.composite_method)
    elif variant.form == 'backtest':  # or
        backtest_signal1(method=variant.method,
                         instruments=variant.instruments,
                         task_id=variant.task_id,
                         period=variant.period,
                         composite_id=variant.composite_id,
                         composite_method=variant.composite_method)

    ## 上面过程调试信号方法，信号参数， 交易方法，交易参数 等确定后 设置到workflow 进行对比
    elif variant.form == 'wfs':  # wf 创建信号值
        auto_workflow(method=variant.method,
                      instruments=variant.instruments,
                      task_id=variant.task_id,
                      period=variant.period,
                      composite_id=variant.composite_id,
                      composite_method=variant.composite_method,
                      model_id=variant.model_id)
    elif variant.form == 'wfm':  # 评估
        metrics_workflow(method=variant.method,
                         instruments=variant.instruments,
                         task_id=variant.task_id,
                         period=variant.period,
                         composite_id=variant.composite_id,
                         composite_method=variant.composite_method,
                         model_id=variant.model_id)

    elif variant.form == 'wfb':  ## 回测
        backtest_workflow(method=variant.method,
                          instruments=variant.instruments,
                          task_id=variant.task_id,
                          period=variant.period,
                          composite_id=variant.composite_id,
                          composite_method=variant.composite_method,
                          model_id=variant.model_id)
