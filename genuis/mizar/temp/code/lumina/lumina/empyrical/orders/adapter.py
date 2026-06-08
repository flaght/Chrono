import datetime, pdb
import pandas as pd
from lumina.empyrical.orders.const import OrderTuple
from lumina.empyrical.orders.pyplot_drawer import plot_his_trade
from lumina.empyrical.orders.echarts_drawer import echarts_his_trade


def generate_orders(orders_data,
                    market_data,
                    direction,
                    code,
                    price_name='close'):
    orders = []
    i = 0
    while i < len(orders_data):
        if orders_data.iloc[i]['trades'] == 1:
            buy_time = orders_data.index[i]
            sell_time = None
            # 寻找平仓信号
            for j in range(i + 1, len(orders_data)):
                if orders_data.iloc[j]['trades'] == -1:
                    sell_time = orders_data.index[j]
                    break
            # 创建订单
            sell_type = 'keep'
            sell_price = None
            if sell_time is not None and direction == 1:
                sell_type = 'win' if market_data.loc[buy_time][
                    price_name] < market_data.loc[sell_time][
                        price_name] else 'loss'
                sell_price = market_data.loc[sell_time][price_name]
            elif sell_time is not None and direction == -1:
                sell_type = 'loss' if market_data.loc[buy_time][
                    price_name] < market_data.loc[sell_time][
                        price_name] else 'win'
                sell_price = market_data.loc[sell_time][price_name]

            orders.append(
                OrderTuple(buy_time=buy_time,
                           buy_price=market_data.loc[buy_time][price_name],
                           sell_time=sell_time,
                           sell_price=sell_price,
                           buy_cnt=1,
                           sell_type=sell_type,
                           expect_direction=direction,
                           buy_symbol=code))
            # 跳过已处理区间
            i = j if sell_time else len(orders_data)
        else:
            i += 1
    return orders


def process_data(orders_data,
                 market_data,
                 direction,
                 code,
                 begin_time=None,
                 end_time=None,
                 time_name='trade_time',
                 price_name='close',
                 time_fmt='%Y-%m-%d %H:%M:%S'):
    market_data.index = market_data[time_name]
    orders = generate_orders(orders_data=orders_data,
                             market_data=market_data,
                             direction=direction,
                             code=code,
                             price_name=price_name)
    pdb.set_trace()
    if begin_time is not None and end_time is not None:
        orders = [
            order for order in orders
            if (order.sell_time is None and order.buy_time <= end_time) or (
                order.buy_time <= end_time and order.sell_time >= begin_time)
        ]
        orders = sorted(orders, key=lambda x: x.buy_time)
        pdb.set_trace()
        end_time = orders[-1].sell_time if orders[
            -1].sell_time else market_data.index[-1].strftime(time_fmt)

        market_data = market_data[
            (market_data.index >= datetime.datetime.strptime(
                orders[0].buy_time, time_fmt) - datetime.timedelta(hours=1))
            & (market_data.index <= datetime.datetime.strptime(
                end_time, time_fmt) + datetime.timedelta(hours=1))]
    market_data = market_data.copy()
    market_data['key'] = list(range(0, len(market_data)))
    return orders, market_data


def plot_orders(orders_data,
                market_data,
                direction,
                code,
                begin_time=None,
                end_time=None,
                y_zoon=1.5,
                time_name='trade_time',
                price_name='close',
                time_fmt='%Y-%m-%d %H:%M:%S',
                file_name=None):
    orders, market_data = process_data(orders_data=orders_data,
                                       market_data=market_data,
                                       direction=direction,
                                       code=code,
                                       begin_time=begin_time,
                                       end_time=end_time,
                                       time_name=time_name,
                                       price_name=price_name,
                                       time_fmt=time_fmt)
    plot_his_trade(kl_pd=market_data,
                   orders=orders,
                   y_zoon=y_zoon,
                   time_fmt=time_fmt,
                   time_name=time_name,
                   price_name=price_name,
                   file_name=file_name)


def echarts_orders(orders_data,
                   market_data,
                   direction,
                   code,
                   begin_time=None,
                   end_time=None,
                   y_zoon=1.5,
                   time_name='trade_time',
                   price_name='close',
                   time_fmt='%Y-%m-%d %H:%M:%S',
                   file_name=None):
    orders, market_data = process_data(orders_data=orders_data,
                                       market_data=market_data,
                                       direction=direction,
                                       code=code,
                                       begin_time=begin_time,
                                       end_time=end_time,
                                       time_name=time_name,
                                       price_name=price_name,
                                       time_fmt=time_fmt)
    line_chart = echarts_his_trade(kl_pd=market_data,
                                   orders=orders,
                                   y_zoon=y_zoon,
                                   time_fmt=time_fmt,
                                   time_name=time_name,
                                   price_name=price_name)
    if isinstance(file_name, str):
        line_chart.render(file_name)
    line_chart.render_notebook()


def plot_orders_file(orders_file,
                     market_file,
                     direction,
                     code,
                     begin_time=None,
                     end_time=None,
                     y_zoon=1.5,
                     price_name='close',
                     time_name='trade_time',
                     fmt='%Y-%m-%d %H:%M:%S',
                     file_name=None):
    orders_data = pd.read_csv(orders_file, index_col=0)
    market_data = pd.read_feather(market_file)

    plot_orders(orders_data=orders_data,
                market_data=market_data,
                direction=direction,
                code=code,
                begin_time=begin_time,
                end_time=end_time,
                y_zoon=y_zoon,
                price_name=price_name,
                time_name=time_name,
                time_fmt=fmt,
                file_name=file_name)
