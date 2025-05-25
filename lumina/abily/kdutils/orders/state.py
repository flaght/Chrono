import numpy as np
import pdb, math


### 订单胜率
def win_rate(orders):
    win_orders = [order for order in orders if order.sell_type == 'win']
    loss_orders = [order for order in orders if order.sell_type == 'loss']
    win_rate = len(win_orders) / (len(win_orders) + len(loss_orders))
    return win_rate


### 订单总盈利
def profit_rate(orders):
    rate = 0
    for order in orders:
        ## 订单已经完成
        if order.sell_type != 'keep':
            if order.expect_direction == 1:
                rate += math.log(
                order.sell_price /
                order.buy_price) - (order.commission + order.slippage) * 2
            else:
                rate += math.log(
                    order.buy_price /
                    order.sell_price) - (order.commission + order.slippage) * 2
                
        else:
            if order.expect_direction == 1:
                rate += math.log(
                    order.sell_price /
                    order.buy_price) - (order.commission + order.slippage)
            else:
                rate += math.log(
                    order.buy_price /
                    order.sell_price) - (order.commission + order.slippage)
    return rate


def pnl(orders):
    pnl = 0
    for order in orders:
        if order.expect_direction == 1:  ## 多头单
            pnl += (order.sell_price - order.buy_price) * order.buy_cnt
        else:  ## 空头单
            pnl += (order.buy_price - order.sell_price) * order.buy_cnt
    return pnl


'''
#每笔盈利交易的平均利润
def avg_win(orders):
    win_orders = [order for order in orders if order.sell_type == 'win']
    if len(win_orders) == 0:
        return 0
    total_win = sum([(order.sell_price - order.buy_price) * order.buy_cnt *
                     order.expect_direction for order in win_orders])
    avg_win = total_win / len(win_orders)
    return avg_win


#每笔亏损交易的平均损失
def avg_loss(orders):
    loss_orders = [order for order in orders if order.sell_type == 'loss']
    if len(loss_orders) == 0:
        return 0
    total_loss = sum([(order.buy_price - order.sell_price) * order.buy_cnt *
                      order.expect_direction for order in loss_orders])
    avg_loss = total_loss / len(loss_orders)
    return avg_loss


#每笔交易的平均利润
def avg_pnl(orders):
    total_pnl = sum([(order.sell_price - order.buy_price) * order.buy_cnt *
                     order.expect_direction for order in orders])
    avg_pnl = total_pnl / len(orders)
    return avg_pnl


# 收益比率
def pay_off(orders):
    avg_win1 = avg_win(orders)
    avg_loss1 = avg_loss(orders)
    pay_off = (avg_win1 / avg_loss1) if avg_loss1 != 0 else float('inf')
    return pay_off


## 每笔盈亏均值 除以标准差
def pnl_std(orders, n_sigma=3):
    pnl = [(order.sell_price - order.buy_price) /  order.buy_price *
           order.expect_direction for order in orders]
    if len(pnl) == 0:
        return 0

    pnl_series = np.array(pnl)
    ## 3倍sigma
    mean_val = np.mean(pnl_series)
    std_val = np.std(pnl_series)
    if std_val == 0:  # 所有数据点都相同，或者只有一个数据点，没有离群值可言
        return pnl_series

    lower_bound = mean_val - n_sigma * std_val
    upper_bound = mean_val + n_sigma * std_val
    pnl_series = pnl_series[(pnl_series >= lower_bound)
                            & (pnl_series <= upper_bound)]
    return pnl_series.mean() / pnl_series.std()
'''
