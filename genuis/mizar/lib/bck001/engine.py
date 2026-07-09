import os, json
import pandas as pd
from pathlib import Path
import multiprocessing as mp

from kdutils.macro2 import base_path
from lib.uvx import *
from lib.cux001 import FactorEvaluate1
from lib.rl012.sandbox import PositionBacktester
from lib.bck001.helpers import *
from lib.bck001.online import TradeOnline1, TradeOnline2

MAPPING_COMPOSITE = {'equal_weight': 'linear', "rl": "model"}

_PARALLEL_MARKET_DATA = None


def _parallel_online(signal_data, name, parts, period, contract_multiplier,
                     params, basic_path):
    print(name)
    if 'value' in signal_data.columns:
        signal_data = signal_data.drop(['value'], axis=1)
    signal_data = signal_data.rename(columns={'transformed': 'value'})
    market_data = _PARALLEL_MARKET_DATA

    if params['category'] == 'online01':
        trade = TradeOnline1(hold_bars=int(period))
    elif params['category'] == 'online02':
        trade = TradeOnline2(hold_bars=int(period))

    res = []
    for signal in signal_data.itertuples():
        rt = trade.on_bar(trade_time=signal.trade_time,
                          code=signal.code,
                          raw_signal=signal.signal)
        res.extend(rt)

    position_data = pd.DataFrame(res)
    pdb.set_trace()
    ## 适配成对应格式
    instruction_data = position_data.copy()
    instruction_data["trade_time"] = pd.to_datetime(
        instruction_data["trade_time"])
    ## extend 不是交易，只是状态续期；numbers=0 也不能进回测
    instruction_data = instruction_data[instruction_data["signal_type"].isin(
        ["open", "close"])].copy()
    instruction_data = instruction_data[instruction_data["numbers"] > 0].copy()

    instruction_data["date"] = instruction_data["trade_time"].dt.normalize()
    instruction_data["min_time"] = instruction_data["trade_time"].dt.strftime(
        "%H%M")

    pb = PositionBacktester(market_data=market_data,
                            contract_multiplier=contract_multiplier,
                            slippage=0.001)
    trade_records, daily_stats = pb.run(position_df=instruction_data,
                                        code='RB')

    dirs1 = os.path.join(basic_path, parts[-3], parts[-2], name)
    os.makedirs(dirs1, exist_ok=True)
    print(dirs1)
    position_data.to_feather(os.path.join(dirs1, "position_data.feather"))
    trade_records.to_feather(os.path.join(dirs1, "trade_records.feather"))
    daily_stats.to_feather(os.path.join(dirs1, "daily_stats.feather"))


def _parallel_backtest(signal_data, name, parts, period, contract_multiplier,
                       params, basic_path):
    print(name)
    if 'value' in signal_data.columns:
        signal_data = signal_data.drop(['value'], axis=1)
    signal_data = signal_data.rename(columns={'transformed': 'value'})
    market_data = _PARALLEL_MARKET_DATA
    position_data = build_paired_position_signals(
        model_output=signal_data,
        hold_bars=period,
        lot_per_signal=params['lot_per_signal'],
        entry_resampling_win=params['entry_resampling_win'],
        #max_active_lots=params['max_active_lots'],
        max_active_lots=params['max_active_open_lots'],
        value_col="value",
        signal_col="signal",
        date_col="trade_date" if "trade_date" in signal_data.columns else None,
        allow_overnight=True,
    )
    pdb.set_trace()
    pb = PositionBacktester(market_data=market_data,
                            contract_multiplier=contract_multiplier,
                            slippage=0.001)
    trade_records, daily_stats = pb.run(position_df=position_data, code='RB')
    dirs1 = os.path.join(basic_path, parts[-3], parts[-2], name)
    os.makedirs(dirs1, exist_ok=True)
    position_data.to_feather(os.path.join(dirs1, "position_data.feather"))
    trade_records.to_feather(os.path.join(dirs1, "trade_records.feather"))
    daily_stats.to_feather(os.path.join(dirs1, "daily_stats.feather"))


def _signal_to_save(data, name, key1, key2, params, base_dirs):
    signal_dt = create_signal(data=data.copy(),
                              signal_method=key1,
                              name='transformed',
                              signal_params=params)
    total_data = signal_dt.merge(data, on=['trade_time', 'code'])
    filename = os.path.join(base_dirs, "{0}_{1}.feather".format(key2, name))
    print("output:{0}".format(filename))
    total_data.to_feather(filename)


def load_er_data1(method,
                  instruments,
                  task_id,
                  period,
                  composite_method,
                  composite_id,
                  val_name,
                  test_name,
                  category='or'):

    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dirs_path = os.path.join(base_path1, "composite",
                             MAPPING_COMPOSITE[composite_method],
                             composite_method, str(composite_id), 'data',
                             category)

    #dirs1 = os.path.join(base_path1, "signal", composite_method)
    val_data = pd.read_feather(
        os.path.join(dirs_path, "{0}.feather").format(val_name))
    test_data = pd.read_feather(
        os.path.join(dirs_path, "{0}.feather").format(test_name))
    return val_data, test_data


def load_er_data2(method,
                  instruments,
                  task_id,
                  period,
                  composite_method,
                  composite_id,
                  val_name,
                  test_name,
                  category='or'):
    pdb.set_trace()
    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dirs_path = os.path.join(base_path1, "composite",
                             MAPPING_COMPOSITE[composite_method],
                             composite_method, str(composite_id), 'data',
                             category)

    val_data = pd.read_feather(
        os.path.join(dirs_path, "{0}.feather").format(val_name))
    test_data = pd.read_feather(
        os.path.join(dirs_path, "{0}.feather").format(test_name))

    val_data1 = pd.read_feather(
        os.path.join(base_path1, "data", "{0}_data.feather".format('val')))
    test_data1 = pd.read_feather(
        os.path.join(base_path1, "data", "{0}_data.feather".format('test')))

    pdb.set_trace()
    val_data['code'] = INSTRUMENTS_CODES[instruments]
    test_data['code'] = INSTRUMENTS_CODES[instruments]
    val_data = val_data.merge(
        val_data1[['trade_time', 'code', 'nxt1_ret_5h']],
        on=['trade_time',
            'code']).rename(columns={'net_er_out': 'transformed'})

    test_data = test_data.merge(
        test_data1[['trade_time', 'code', 'nxt1_ret_5h']],
        on=['trade_time',
            'code']).rename(columns={'net_er_out': 'transformed'})

    # val_data = val_data.drop(['signal'], axis=1).merge(
    #     val_data1[['trade_time', 'code', 'nxt1_ret_5h']],
    #     on=['trade_time',
    #         'code']).rename(columns={'net_er_out': 'transformed'})

    # test_data = test_data.drop(['signal'], axis=1).merge(
    #     test_data1[['trade_time', 'code', 'nxt1_ret_5h']],
    #     on=['trade_time',
    #         'code']).rename(columns={'net_er_out': 'transformed'})

    return val_data, test_data


def create_composite_signal(method, instruments, task_id, period,
                            composite_method, composite_id, signal_functions,
                            val_data, test_data):
    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dirs1 = os.path.join(base_path1, "signal", "proto", composite_method)

    for key1, functions in signal_functions.items():
        for key1, functions in signal_functions.items():
            for key2, params in functions.items():
                print(key1, key2)
                base_dirs = os.path.join(dirs1, str(composite_id), key1)
                os.makedirs(base_dirs, exist_ok=True)
                _signal_to_save(data=test_data,
                                name='test',
                                key1=key1,
                                key2=key2,
                                base_dirs=base_dirs,
                                params=params)
                _signal_to_save(data=val_data,
                                name='val',
                                key1=key1,
                                key2=key2,
                                base_dirs=base_dirs,
                                params=params)


def metrics_composite_signal(method, instruments, task_id, period,
                             composite_method, composite_id):
    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dirs1 = os.path.join(base_path1, "signal", "proto", composite_method,
                         str(composite_id))
    file_path = Path(dirs1)
    
    for feat_file in file_path.rglob('*.feather'):
        signal_data = pd.read_feather(feat_file)
        name = feat_file.parts[-1].split('.')[0]
        evaluate = FactorEvaluate1(factor_data=signal_data,
                                   factor_name='signal',
                                   ret_name='nxt1_ret_{0}h'.format(period),
                                   roll_win=15,
                                   fee=0.0,
                                   scale_method='raw',
                                   expression="{0}_{1}_{2}_{3}".format(
                                       feat_file.parts[-4],
                                       feat_file.parts[-3],
                                       feat_file.parts[-2], name),
                                   resampling_win=period,
                                   name=name)
        output = os.path.join(feat_file.parent, "metrics")
        os.makedirs(output, exist_ok=True)
        _ = evaluate.run()
        evaluate.plot_results()
        evaluate.save_results(output)


def backtest_composite_signal(method, instruments, task_id, period,
                              composite_method, composite_id, params,
                              trading_sessions):
    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dirs1 = os.path.join(base_path1, "signal", "proto", composite_method,
                         str(composite_id)) ## 信号值

    min_time = None
    max_time = None
    res = []
    file_path = Path(dirs1)
    pdb.set_trace()
    for feat_file in file_path.rglob('*.feather'):
        print(feat_file)
        signal_data = pd.read_feather(feat_file)
        name = feat_file.parts[-1].split('.')[0]
        if 'test' not in name:
            continue
        parts = feat_file.parts
        min_time = signal_data['trade_time'].min(
        ) if min_time is None else min(signal_data['trade_time'].min(),
                                       min_time)
        max_time = signal_data['trade_time'].max(
        ) if max_time is None else max(signal_data['trade_time'].max(),
                                       max_time)
        res.append((name, parts, signal_data))
    pdb.set_trace()
    market_data = load_market_data(instruments=instruments,
                                   begin_time=min_time,
                                   end_time=max_time,
                                   trading_sessions=trading_sessions)
    global _PARALLEL_MARKET_DATA
    _PARALLEL_MARKET_DATA = market_data

    basic_path = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl', 'backtest',
                              "proto", composite_method,
                              Params.create_tag(params)) # 保存回测路劲

    os.makedirs(basic_path, exist_ok=True)
    config_path = os.path.join(basic_path, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    args_for_backtest = [(item[2], item[0], item[1], period, 10, params,
                          basic_path) for item in res]

    if len(res) <= 1:
        res = [
            _parallel_backtest(signal_data=args_for_backtest[0][0],
                               name=args_for_backtest[0][1],
                               parts=args_for_backtest[0][2],
                               period=args_for_backtest[0][3],
                               contract_multiplier=args_for_backtest[0][4],
                               params=args_for_backtest[0][5],
                               basic_path=args_for_backtest[0][6])
        ]
    else:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()
        with ctx.Pool(processes=4) as pool:
            res = pool.starmap(_parallel_backtest, args_for_backtest)
    print(Params.create_tag(params))


def online_composite_signal(method, instruments, task_id, period,
                            composite_method, composite_id, params,
                            trading_sessions):
    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dirs1 = os.path.join(base_path1, "signal", composite_method,
                         str(composite_id))

    min_time = None
    max_time = None
    res = []
    file_path = Path(dirs1)
    for feat_file in file_path.rglob('*.feather'):
        print(feat_file)
        signal_data = pd.read_feather(feat_file)
        name = feat_file.parts[-1].split('.')[0]
        if 'test' not in name:
            continue
        parts = feat_file.parts
        min_time = signal_data['trade_time'].min(
        ) if min_time is None else min(signal_data['trade_time'].min(),
                                       min_time)
        max_time = signal_data['trade_time'].max(
        ) if max_time is None else max(signal_data['trade_time'].max(),
                                       max_time)
        res.append((name, parts, signal_data))

    market_data = load_market_data(instruments=instruments,
                                   begin_time=min_time,
                                   end_time=max_time,
                                   trading_sessions=trading_sessions)
    global _PARALLEL_MARKET_DATA
    _PARALLEL_MARKET_DATA = market_data
    args_for_backtest = []

    for online in ['online02']:
        params['category'] = online
        basic_path = os.path.join(base_path, method,
                                  instruments, 'temp', 'model', str(task_id),
                                  str(period), 'rl', 'online',
                                  composite_method, online)
        os.makedirs(basic_path, exist_ok=True)
        args_for_backtest += [(item[2], item[0], item[1], period, 10, params,
                               basic_path) for item in res]

    if len(res) <= 1:
        res = [
            _parallel_online(signal_data=args_for_backtest[0][0],
                             name=args_for_backtest[0][1],
                             parts=args_for_backtest[0][2],
                             period=args_for_backtest[0][3],
                             contract_multiplier=args_for_backtest[0][4],
                             params=args_for_backtest[0][5],
                             basic_path=args_for_backtest[0][6])
        ]
    else:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()
        with ctx.Pool(processes=4) as pool:
            res = pool.starmap(_parallel_backtest, args_for_backtest)
    print(Params.create_tag(params))
