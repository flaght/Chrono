import json
from dotenv import load_dotenv
import multiprocessing as mp

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.bck001.engine import *
from lib.bck001.common import *
from chaosmind.timing.sirius0003.workflow import WorkFlow

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


def _workflow(data, factors_infos, instruments, model_id, period, params, name,
              dirs1):
    pdb.set_trace()
    wf = WorkFlow(directory=params['model_path'],
                  factors_infos=factors_infos,
                  code=INSTRUMENTS_CODES[instruments],
                  symbol="{0}9999".format(instruments.lower()),
                  task_id=model_id,
                  period=period,
                  softmax_temperature=params['softmax_temperature'],
                  min_open_signal_abs=params['min_open_signal_abs'],
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
    #     #     break
    #     res1.append(signal)
    #     res2.extend(events)

    # signal_data = pd.DataFrame(res1)
    # signal_data = signal_data.merge(
    #     data[['trade_time', 'code', 'nxt1_ret_{}h'.format(period)]],
    #     on=['trade_time', 'code'])
    # filename = os.path.join(dirs1, "{0}.feather".format(name))
    # signal_data.to_feather(filename)

    # events_data = pd.DataFrame(res2)

    # ## 全保留，在绩效时候 再做相应处理
    # #events_data = events_data[events_data['signal_type'] != 'close']

    # events_data = events_data.merge(
    #     data[['trade_time', 'code', 'nxt1_ret_{}h'.format(period)]],
    #     on=['trade_time', 'code'])
    # filename = os.path.join(dirs1, "event_{0}.feather".format(name))
    # events_data.to_feather(filename)


def online_signal1(method, instruments, task_id, period, composite_method,
                   composite_id):
    trading_sessions = (("21:00", "23:00"), ("09:00", "10:15"),
                        ("10:30", "11:30"), ("13:30", "15:00"))
    params = {}

    online_composite_signal(method=method,
                            instruments=instruments,
                            task_id=task_id,
                            period=period,
                            composite_method=composite_method,
                            composite_id=composite_id,
                            params=params,
                            trading_sessions=trading_sessions)


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
    val_data, test_data = load_er_data2(method=method,
                                        instruments=instruments,
                                        task_id=task_id,
                                        period=period,
                                        composite_method=composite_method,
                                        composite_id=composite_id,
                                        val_name='val_data',
                                        test_name='test_data',
                                        category='or')
    create_composite_signal(method=method,
                            instruments=instruments,
                            task_id=task_id,
                            period=period,
                            composite_method=composite_method,
                            composite_id=composite_id,
                            signal_functions=signal_functions,
                            val_data=val_data,
                            test_data=test_data)


def auto_workflow(method, instruments, task_id, period, composite_method,
                  composite_id, model_id):
    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dirs1 = os.path.join(base_path1, "signal", "final", composite_method,
                         model_id, "data")
    os.makedirs(dirs1, exist_ok=True)

    factors_infos, params = load_sirius_params(
        code=INSTRUMENTS_CODES[instruments], task_id=str(model_id))

    val_data, test_data = load_er_data2(method=method,
                                        instruments=instruments,
                                        task_id=task_id,
                                        period=period,
                                        composite_method=composite_method,
                                        composite_id=composite_id,
                                        val_name='val_data',
                                        test_name='test_data',
                                        category='wf')

    _workflow(data=val_data,
              factors_infos=factors_infos,
              instruments=instruments,
              model_id=model_id,
              period=period,
              params=params,
              name='val',
              dirs1=dirs1)

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
                      model_id=variant.composite_id)

    elif variant.form == 'wfm':  # 评估
        metrics_workflow(method=variant.method,
                         instruments=variant.instruments,
                         task_id=variant.task_id,
                         period=variant.period,
                         composite_id=variant.composite_id,
                         composite_method=variant.composite_method,
                         model_id=variant.composite_id)

    elif variant.form == 'wfb':  ## 回测
        backtest_workflow(method=variant.method,
                          instruments=variant.instruments,
                          task_id=variant.task_id,
                          period=variant.period,
                          composite_id=variant.composite_id,
                          composite_method=variant.composite_method,
                          model_id=variant.composite_id)
        
    elif variant.form == 'online':
        online_signal1(method=variant.method,
                       instruments=variant.instruments,
                       task_id=variant.task_id,
                       period=variant.period,
                       composite_id=variant.composite_id,
                       composite_method=variant.composite_method)
