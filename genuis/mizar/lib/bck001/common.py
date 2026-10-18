import os
import pdb
import pandas as pd
from lib.cux001 import FactorEvaluate1
from lib.bck002 import load_market_data
from lib.rl012.sandbox import PositionBacktester


def workflow_creator(wf, data, params, period, name, dirs1):
    res1 = []
    res2 = []
    i = 0
    wf.initialization() ## 在这个过程中，不做状态持久化存储
    for i in range(0, data.shape[0]):
        if i <= params['signal_params']['roll_num']:
            continue
        raw_action = data.loc[i - params['signal_params']['roll_num']:i]
        trade_time = raw_action.loc[i]['trade_time']
        print(trade_time)
        signal, events = wf.wrapper(trade_time=trade_time,
                                    raw_action=raw_action,
                                    name='transformed')
        # pdb.set_trace()
        # if i > 2000:
        #     break
        res1.append(signal)
        res2.extend(events)

    signal_data = pd.DataFrame(res1)
    signal_data = signal_data.merge(
        data[['trade_time', 'code', 'nxt1_ret_{}h'.format(period)]],
        on=['trade_time', 'code'])
    filename = os.path.join(dirs1, "{0}.feather".format(name))
    signal_data.to_feather(filename)

    events_data = pd.DataFrame(res2)

    ## 全保留，在绩效时候 再做相应处理
    #events_data = events_data[events_data['signal_type'] != 'close']

    events_data = events_data.merge(
        data[['trade_time', 'code', 'nxt1_ret_{}h'.format(period)]],
        on=['trade_time', 'code'])
    filename = os.path.join(dirs1, "event_{0}.feather".format(name))
    events_data.to_feather(filename)


def metrics_event(base_dir, period, name, roll_win=15, fee=0.00001):
    pdb.set_trace()
    filename = os.path.join(os.path.join(base_dir, "data"),
                            "event_{0}.feather".format(name))
    event_data = pd.read_feather(filename)
    output = os.path.join(base_dir, "metrics")
    evaluate = FactorEvaluate1(
        factor_data=event_data,
        factor_name='position_direction',
        ret_name='nxt1_ret_{0}h'.format(period),
        roll_win=roll_win,
        fee=fee,
        scale_method='raw',
        expression="final_{0}".format(name),
        resampling_win=1,  #period, 事件产生的开仓信号到平仓，持仓5分钟，当前可以用于连续开仓。模拟回测效果
        name="event_{0}".format(name))
    os.makedirs(output, exist_ok=True)
    _ = evaluate.run()
    evaluate.plot_results()
    evaluate.save_results(output)


def metrics_signal(base_dir, period, name, roll_win=15, fee=0.0):
    filename = os.path.join(os.path.join(base_dir, "data"),
                            "{0}.feather".format(name))
    signal_data = pd.read_feather(filename)
    output = os.path.join(base_dir, "metrics")
    evaluate = FactorEvaluate1(factor_data=signal_data,
                               factor_name='transformed',
                               ret_name='nxt1_ret_{0}h'.format(period),
                               roll_win=roll_win,
                               fee=0.0,
                               scale_method='raw',
                               expression="final_{0}".format(name),
                               resampling_win=period,
                               name=name)
    os.makedirs(output, exist_ok=True)
    _ = evaluate.run()
    evaluate.plot_results()
    evaluate.save_results(output)



def backtest_event(instruments, period, name, base_dir, output_dir):
    pdb.set_trace()
    trading_sessions = (("21:00", "23:00"), ("09:00", "10:15"),
                        ("10:30", "11:30"), ("13:30", "15:00"))

    filename = os.path.join(os.path.join(base_dir, "data"),
                            "event_{0}.feather".format(name))
    event_data = pd.read_feather(filename)
    pdb.set_trace()
    min_time = event_data['trade_time'].min()
    max_time = event_data['trade_time'].max()
    market_data = load_market_data(instruments=instruments,
                                   begin_time=min_time,
                                   end_time=max_time,
                                   trading_sessions=trading_sessions)

    event_data = event_data[[
        'trade_time', 'code', 'direction', 'numbers', 'signal_type'
    ]].copy()

    event_data = event_data[event_data['signal_type'].isin(['open',
                                                            'close'])].copy()
    event_data['trade_time'] = pd.to_datetime(event_data['trade_time'])
    event_data['date'] = event_data['trade_time'].dt.normalize()
    event_data['min_time'] = event_data['trade_time'].dt.strftime('%H%M')

    pb = PositionBacktester(market_data=market_data,
                            contract_multiplier=10,
                            slippage=0.001)
    trade_records, daily_stats = pb.run(position_df=event_data, code='RB')
    dirs1 = os.path.join(output_dir, name)
    os.makedirs(dirs1, exist_ok=True)
    event_data.reset_index(drop=True).to_feather(
        os.path.join(dirs1, "position_data.feather"))
    trade_records.to_feather(os.path.join(dirs1, "trade_records.feather"))
    daily_stats.to_feather(os.path.join(dirs1, "daily_stats.feather"))