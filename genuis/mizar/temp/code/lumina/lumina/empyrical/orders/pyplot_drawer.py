from ultron.kdutils.progress import Progress
from ultron.utilities.logger import kd_logger
from ultron.kdutils.date import str_to_datetime
import pandas as pd
import matplotlib.pyplot as plt
import pdb


def plot_his_trade(orders,
                   kl_pd,
                   y_zoon=1.5,
                   time_fmt='%Y-%m-%d %H:%M:%S',
                   time_name='trade_time',
                   price_name='close',
                   file_name=None):
    """
    可视化绘制Order对象，绘制交易买入时间，卖出时间，价格等
    :param orders: Order对象序列
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :return:
    """
    # 拿出时间序列中最后一个，做为当前价格
    now_price = kl_pd.iloc[-1][price_name]
    all_pd = kl_pd

    ax_cnt = 1.5 * len(orders)
    fig, ax = plt.subplots(figsize=(14 + ax_cnt, 8 * y_zoon))
    times = [time.strftime(time_fmt) for time in all_pd.index]

    ## 压缩Y轴

    min_close = all_pd[price_name].min()
    max_close = all_pd[price_name].max()
    padding = (max_close - min_close) * 0.1

    # Add slightly more padding specifically for annotations if needed
    annotation_padding_factor = 0.05  # Add 5% extra range for annotations
    ylim_min = min_close - padding - (max_close -
                                      min_close) * annotation_padding_factor
    ylim_max = max_close + padding + (max_close -
                                      min_close) * annotation_padding_factor
    y_range = ylim_max - ylim_min  # Calculate the full visible y-range

    ax.set_ylim(ylim_min, ylim_max)

    ### 绘制当前价格线
    ax.plot(times, all_pd[price_name])
    # 填充透明blue, 针对用户一些版本兼容问题进行处理
    ax.fill_between(times, 0, all_pd[price_name], color='blue', alpha=.18)

    ### X轴时间格式化
    step = 10
    ax.set_xticks(range(0, len(times), step))
    ax.set_xticklabels(times[::step], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    with Progress(len(orders), 0) as pg:
        for index, order in enumerate(orders):
            pg.show(index + 1)
            mask_time = all_pd[time_name] == order.buy_time
            st_key = all_pd[mask_time]['key']
            if order.sell_type == 'keep':
                rv_pd = all_pd.iloc[st_key.values[0]:, :]
            else:
                mask_sell_time = all_pd[time_name] == order.sell_time
                st_sell_key = all_pd[mask_sell_time]['key']
                rv_pd = all_pd.iloc[st_key.values[0]:st_sell_key.values[0] +
                                    1, :]

            try:
                if order.sell_type == 'keep':
                    order_win = (now_price -
                                 order.buy_price) * order.expect_direction > 0
                elif order.sell_type == 'win':
                    order_win = True
                else:
                    order_win = False
                if order_win:
                    # 盈利的使用红色
                    plt.fill_between(
                        [time.strftime(time_fmt) for time in rv_pd.index],
                        0,
                        rv_pd[price_name],
                        color='red',
                        alpha=.18)
                else:
                    # 亏损的使用绿色
                    plt.fill_between(
                        [time.strftime(time_fmt) for time in rv_pd.index],
                        0,
                        rv_pd[price_name],
                        color='green',
                        alpha=.38)
            except:
                kd_logger.warning('fill_between numpy type not safe!')

            # 格式化买入信息标签
            #buy_time_fmt = str_to_datetime(str(order.buy_time), fmt)
            buy_tip = '{:.2f}'.format(order.buy_price)

            #st_key.values[0]
            # 写买入tip信息
            buy_time_dt_for_plot = all_pd.index[st_key.values[0]]
            buy_price_at_time = all_pd.loc[buy_time_dt_for_plot, price_name]
            y_offset_factor = 0.03
            y_buy_text_pos = buy_price_at_time + y_range * y_offset_factor
            y_buy_text_pos = max(
                ylim_min + y_range * 0.01,
                min(ylim_max - y_range * 0.01, y_buy_text_pos))

            plt.annotate(buy_tip,
                         xy=(buy_time_dt_for_plot.strftime(time_fmt),
                             buy_price_at_time),
                         xytext=(buy_time_dt_for_plot.strftime(time_fmt),
                                 y_buy_text_pos),
                         arrowprops=dict(facecolor='red'),
                         horizontalalignment='center',
                         verticalalignment='bottom')
            '''
            plt.annotate(buy_tip,
                         xy=(buy_time_fmt.strftime(fmt),
                             all_pd['close'].asof(buy_time_fmt) * 2 / 5),
                         xytext=(buy_time_fmt.strftime(fmt),
                                 all_pd['close'].asof(buy_time_fmt)),
                         arrowprops=dict(facecolor='red'),
                         horizontalalignment='left',
                         verticalalignment='top')
            '''

            if order.sell_price is not None:
                sell_time_dt_for_plot = rv_pd.index[-1]
                sell_price_at_time = all_pd.loc[sell_time_dt_for_plot,
                                                price_name]

                pfr = (order.sell_price - order.buy_price) / order.buy_price
                sell_tip = '\n\n{:.2f}\n{:.2f}%'.format(
                    order.sell_price, pfr * 100)
            else:
                # 如果单子未卖出，卖出入信息标签使用，收益使用now_price计算，需＊单子期望的盈利方向
                sell_time_dt_for_plot = all_pd.index[-1]
                sell_price_at_time = now_price
                pfr = (now_price - order.buy_price) / order.buy_price
                sell_tip = '\n\n{:.2f}\n{:.2f}%'.format(now_price, pfr * 100)

            y_sell_text_pos = sell_price_at_time + y_range * y_offset_factor
            y_sell_text_pos = max(
                ylim_min + y_range * 0.01,
                min(ylim_max - y_range * 0.01, y_sell_text_pos))

            plt.annotate(sell_tip,
                         xy=(sell_time_dt_for_plot.strftime(time_fmt),
                             sell_price_at_time),
                         xytext=(sell_time_dt_for_plot.strftime(time_fmt),
                                 y_sell_text_pos),
                         arrowprops=dict(facecolor='green'),
                         horizontalalignment='center',
                         verticalalignment='bottom')
            '''
            if order.sell_price is not None:
                # 如果单子卖出，卖出入信息标签使用，收益使用sell_price计算，需＊单子期望的盈利方向
                sell_time_fmt = str_to_datetime(str(order.sell_time), fmt)
                #pft = (order.sell_price - order.buy_price) * order.buy_cnt * order.expect_direction
                pfr = (order.sell_price - order.buy_price) / order.buy_price
                sell_tip = '\n\n{:.2f}\n{:.2f}%'.format(order.sell_price, pfr * 100)
            else:
                # 如果单子未卖出，卖出入信息标签使用，收益使用now_price计算，需＊单子期望的盈利方向
                sell_time_fmt = str_to_datetime(
                    str(all_pd[-1:]['trade_time'][0]), fmt)
                #pft = (now_price - order.buy_price) * order.buy_cnt * order.expect_direction
                pfr = (now_price - order.buy_price) / order.buy_price
                sell_tip = '\n\n{:.2f}\n{:.2f}%'.format(now_price, pfr * 100)

            # 写卖出tip信息
            plt.annotate(sell_tip,
                         xy=(sell_time_fmt.strftime(fmt),
                             all_pd['close'].asof(sell_time_fmt) * 2 / 5),
                         xytext=(sell_time_fmt.strftime(fmt),
                                 all_pd['close'].asof(sell_time_fmt)),
                         arrowprops=dict(facecolor='green'),
                         horizontalalignment='left',
                         verticalalignment='top')
            '''

    plt.title(order.buy_symbol)
    if isinstance(file_name, str):
        plt.savefig(file_name)
