import pdb,datetime
import pandas as pd
from lumina.orders.const import OrderTuple
from lumina.orders.drawer import plot_his_trade


def generate_orders(trader_data, market_data, direction, code, key='close'):
    orders = []
    i = 0
    while i < len(trader_data):
        if trader_data.iloc[i]['trades'] == 1:
            buy_time = trader_data.index[i]
            sell_time = None
            # 寻找平仓信号
            for j in range(i + 1, len(trader_data)):
                if trader_data.iloc[j]['trades'] == -1:
                    sell_time = trader_data.index[j]
                    break
            # 创建订单
            sell_type = 'keep'
            sell_price = None
            if sell_time is not None and direction == 1:
                sell_type = 'win' if market_data.loc[buy_time][
                    key] < market_data.loc[sell_time][key] else 'loss'
                sell_price = market_data.loc[sell_time][key]
            elif sell_time is not None and direction == -1:
                sell_type = 'loss' if market_data.loc[buy_time][
                    key] < market_data.loc[sell_time][key] else 'win'
                sell_price = market_data.loc[sell_time][key]

            orders.append(
                OrderTuple(buy_time=buy_time,
                           buy_price=market_data.loc[buy_time][key],
                           sell_time=sell_time,
                           sell_price=sell_price,
                           buy_cnt=1,
                           sell_type=sell_type,
                           expect_direction=direction,
                           buy_symbol=code))
            # 跳过已处理区间
            i = j if sell_time else len(trader_data)
        else:
            i += 1
    return orders


def plot_split_orders(trader_file,
                      market_file,
                      direction,
                      code,
                      y_zoon=1.5,
                      begin_date=None,
                      end_date=None):
    trader_data = pd.read_csv(trader_file, index_col=0)
    market_data = pd.read_feather(market_file)
    market_data.index = market_data['trade_time']
    market_data = market_data.rename(columns={'price': 'close'})
    orders = generate_orders(trader_data, market_data, direction, code)
    if begin_date is not None and end_date is not None:
        orders = [
            order for order in orders
            if (order.buy_time <= end_date and order.sell_time >= begin_date)
        ]
        orders = sorted(orders, key=lambda x: x.buy_time)
        market_data = market_data[
            (market_data.index >= datetime.datetime.strptime(
                orders[0].buy_time, '%Y-%m-%d %H:%M:%S'))
            & (market_data.index <= datetime.datetime.strptime(
                orders[-1].sell_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=60))]
    market_data['key'] = list(range(0, len(market_data)))
    #orders = generate_orders(trader_data, market_data, direction, code)
    plot_his_trade(kl_pd=market_data, orders=orders, y_zoon=y_zoon)


def plot_his_orders(trader_file,
                market_file,
                direction,
                code,
                y_zoon=1.5,
                begin_date=None,
                end_date=None):
    trader_data = pd.read_csv(trader_file, index_col=0)
    market_data = pd.read_feather(market_file)
    market_data.index = market_data['trade_time']
    market_data = market_data.rename(columns={'price': 'close'})

    if begin_date is not None and end_date is not None:
        trader_data = trader_data[(trader_data.index >= begin_date)
                                  & (trader_data.index <= end_date)]
        market_data = market_data[(market_data.index >= begin_date)
                                  & (market_data.index <= end_date)]
    market_data['key'] = list(range(0, len(market_data)))
    orders = generate_orders(trader_data, market_data, direction, code)
    plot_his_trade(kl_pd=market_data, orders=orders, y_zoon=y_zoon)


'''
trader_file = '/workspace/worker/pj/Chrono/records/phecda/files/hedge041_trader/trader/eval/3/trader_values_eval_trader.csv'
market_file = '/workspace/data/dev/kd/evolution/nn/phecda/aicso3/normal/ims/rolling/normal_factors3/o2o_1/5_60_5_1_0/normal_factors_test_47.feather'

plot_split_orders(trader_file=trader_file,
            market_file=market_file,
            direction=1, code='IM',
            begin_date='2025-01-20',
            end_date='2025-01-21')
'''
