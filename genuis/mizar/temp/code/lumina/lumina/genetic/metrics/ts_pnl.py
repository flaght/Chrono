import pdb
import numpy as np
import pandas as pd
from lumina.genetic.metrics.empyrical import empyrical
from lumina.genetic.metrics.ksplit import time_series
'''

pos_data = pos_data.shift(1).fillna(0)

change = np.log(openD / openD.shift(1)).shift(-1).fillna(0)

openD / openD.shift(1): 计算当前周期收盘价相对于上一周期收盘价的比率。
np.log(...): 计算自然对数，得到对数收益率。
.shift(-1): 非常关键。将计算出的收益率序列向上移动一行。这意味着在时间点 t 的 change 值，实际上是时间点 t 到 t+1 的价格变动（收益率）。

结合 pos_data.shift(1) 的理解:
在时间点 t，我们持有的仓位 (pos) 是由 t-1 的信号决定的。
在时间点 t，我们计算的收益 (change) 是从 t 到 t+1 发生的。
所以，pos * change 计算的是：用 t-1 的信号在 t 时刻建立的仓位，在 t 到 t+1 这个持有期内产生的收益。这是一种常见的计算方式，即 PnL 确认在持有期结束时。
.fillna(0): 填充 shift(-1) 在最后一行产生的 NaN。

'''


def calculate_ful_ts_ret(pos_data,
                         total_data,
                         strategy_settings,
                         price_name='close',
                         ret_name=None,
                         agg=True) -> pd.DataFrame:
    name = ret_name if isinstance(ret_name, str) else price_name
    if not isinstance(pos_data.columns, pd.MultiIndex):
        new_columns = pd.MultiIndex.from_tuples([("pos", col)
                                                 for col in pos_data.columns])
        pos_data.columns = new_columns

    if not isinstance(total_data.columns, pd.MultiIndex):
        new_columns = pd.MultiIndex.from_tuples([
            ("close", col) for col in total_data.columns
        ])
        total_data.columns = new_columns

    commission_mode = strategy_settings.get('mode', 'rate')  ## 费率机制
    commission_rate = strategy_settings.get('commission',
                                            0.0)  ## 费率:包含费率制和固定费率
    slippage_point = strategy_settings.get('slippage', 0.0)
    '''
    pos_data.loc[t] 代表的是在时间点 t 根据当前及之前所有可用信息计算出来的 目标持仓信号.
    pos_data.shift(1) 这个操作会将 pos_data 中的所有数据向下平移一行。
     1. 原来在 pos_data.loc[t-1] 的信号值，现在会出现在新 pos_data.loc[t] 的位置。
     2. 原来在 pos_data.loc[t] 的信号值，现在会出现在新 pos_data.loc[t+1] 的位置。
     3. 经过 .shift(1) 之后，在时间点 t，我们实际用于计算 PnL 或进行其他操作的持仓 (pos_data.loc[t])，实际上是原始信号在 t-1 时刻的值。
    
    模拟交易执行的延迟 
      1. 策略逻辑计算出一个交易信号（比如在 t-1 时刻的收盘后，或在 t-1 这一分钟内）到实际下单并成交，总会有一个时间差。不可能在信号产生的完全同一瞬间就完成交易。
        .shift(1) 模拟了这种最基本的延迟
      2. 在 t-1 时刻（t-1 分钟的收盘价出来后，或者在 t-1 分钟内）策略产生了新的目标持仓信号 original_pos_data.loc[t-1]。
      3. 会在下一个时间单位的开始，即 t 时刻（例如，t 分钟的开盘时），去执行这个交易，使得我们在 t 时刻的实际持仓变为 original_pos_data.loc[t-1]。

    因果关系：
      1.交易决策（信号）发生在先，市场价格变动和由此产生的盈亏发生在后
      2.original_pos_data.loc[t-1] (决策/信号) -> pos.loc[t] (在t时刻的持仓状态) -> change.loc[t] (t到t+1的价格变动) -> pnl.loc[t] (在t时刻确认的、对应t到t+1持有期的盈亏)
      3.shift(1) 帮助维持了这种正确的因果链条。
    '''
    # --- 准备仓位和价格数据 ---
    pos_data = pos_data.shift(1).fillna(0)
    codes = pos_data.columns.get_level_values(1).unique().tolist()

    df = total_data.loc[:, (name, codes)]

    # --- 计算仓位和收益率 ---
    # 使用 outer join 确保所有日期都包含，然后填充
    df = df.join(pos_data, how='left').fillna(0)
    pos = df['pos']
    pos = pos.fillna(0)

    df2 = total_data.loc[:, (name, codes)]
    openD = df2[name]
    if ret_name is None:
        change = np.log(openD / openD.shift(1)).shift(-1).fillna(
            0)  # 当前期 看到 当前期到下一期的收益率
    else:
        change = total_data[ret_name]

    trade = (pos.diff()).fillna(0)

    if commission_mode == 'rate':
        commission = abs(trade) * commission_rate
    elif commission_mode == 'fixed':
        # 将仓位权重的变化，转换为交易手数的变化
        # trade_hands = trade_weight * (总资金 / 每手所需资金)
        # 可以认为 trade_weight * capital_per_hand 代表了交易的名义价值变化
        # 更直接的是，认为pos=1代表交易1手所需的名义资金，即capital_per_hand
        # 那么 pos.diff() 直接代表了交易的手数比例
        trade_weight = trade
        trade_hands_ratio = abs(trade_weight)  # 交易手数与“标准手”的比例
        commission_amount = trade_hands_ratio * commission_rate  ## commission_rate 表示固定手续费
        #commission_in_return = commission_amount / strategy_settings.get('hand', 0.0) # 交易一手需要占用的名义资金
        # 将手续费金额，转换回“收益率”的维度 (核心！)
        # 收益率扣减 = 手续费金额 / 资产名义价值
        # 资产名义价值 = 价格 * 合约乘数 * 每单位权重对应手数
        # 我们用交易发生时的价格来计算名义价值，更准确
        asset_nominal_value = df[name] * strategy_settings.get('size', 0.0) * 1
        # 加上一个很小的值避免除以零
        commission_in_return = commission_amount / (asset_nominal_value +
                                                    1e-10)
        commission = commission_in_return

    slippage = abs(trade) * slippage_point
    a_ret = change * pos - commission - slippage
    n_ret = change * pos
    df = pd.concat([a_ret, n_ret], axis=1)
    df.columns = ['a_ret', 'n_ret']
    if agg:
        df = df.resample('1D').agg({
            'a_ret': 'sum',
            'n_ret': 'sum',
        }).dropna().fillna({
            'a_ret': 0,
            'n_ret': 0,
        }).fillna(method='ffill')
    return df


def calculate_ful_ts_pnl(pos_data,
                         total_data,
                         strategy_settings,
                         agg=True) -> pd.DataFrame:

    if not isinstance(pos_data.columns, pd.MultiIndex):
        new_columns = pd.MultiIndex.from_tuples([("pos", col)
                                                 for col in pos_data.columns])
        pos_data.columns = new_columns

    if not isinstance(total_data.columns, pd.MultiIndex):
        new_columns = pd.MultiIndex.from_tuples([
            ("close", col) for col in total_data.columns
        ])
        total_data.columns = new_columns
    pos_data = pos_data.shift(1).fillna(0)
    capital = strategy_settings.get('capital', 10000000)
    commission_rate = strategy_settings.get('commission', 0.0)
    if commission_rate > 0.1:
        commission_rate = commission_rate / 10000
    slippage_point = np.array(strategy_settings.get('slippage', 0.0))
    size = np.array(strategy_settings.get('size', 10))

    codes = pos_data.columns.get_level_values(1).unique().tolist()
    #size_list = [strategy_settings.get('size', 10)[code] for code in codes]
    #size_list = [10 for code in codes]
    #size = np.array(size_list).reshape(1, -1)
    df = total_data.loc[:, ('close', codes)]
    df = df.join(pos_data, how='left').fillna(0)
    close = df['close']
    trade_vol = total_data.loc[:, ('trade_vol', codes)]
    trade_vol = trade_vol.join(pos_data, how='left').fillna(0)
    pos = df['pos'] * trade_vol['trade_vol']
    pos = pos.fillna(0)
    df2 = total_data.loc[:, ('open', codes)]
    openD = df2['open']
    change = (openD.diff(1)).shift(-1).fillna(0)  # 修改

    #change = (close.diff()).fillna(0)
    trade = (pos.diff()).fillna(0)
    commission = abs(trade) * close * commission_rate * size
    slippage = abs(trade) * slippage_point * size
    pnl = change * pos * size - commission - slippage

    ret = pnl / capital
    balance = pnl.shape[1] * capital + pnl.sum(1).cumsum()
    balance = balance.to_frame('balance')

    pnl = pnl.sum(1).to_frame('pnl')
    ret = ret.sum(1).to_frame('ret')
    drawdown = (balance -
                balance.cummax()).rename(columns={'balance': 'drawdown'})

    df = pd.concat([balance, drawdown, pnl, ret], axis=1)
    if agg:
        df = df.resample('1D').agg({
            'balance': 'last',
            'drawdown': 'last',
            'pnl': 'sum',
            'ret': 'sum',
        }).dropna().fillna({
            'pnl': 0,
            'ret': 0,
        }).fillna(method='ffill')
    else:
        df = df.dropna().fillna({
            'pnl': 0,
            'ret': 0,
        }).fillna(method='ffill')

    #df['ret'] = df['pnl'] / capital / len(codes) * 2
    df['ret'] = df['pnl'] / capital
    return df


def calculate_ts_ret_metrics(pos_data,
                             total_data,
                             strategy_settings,
                             name='close',
                             agg=True):

    ## 绩效使用配置在strategy_settings
    df = calculate_ful_ts_ret(pos_data=pos_data,
                              total_data=total_data,
                              strategy_settings=strategy_settings,
                              name=name,
                              agg=agg)
    evaluate_params = strategy_settings['evaluate_params']
    method = evaluate_params['method']
    params = {} if 'params' not in evaluate_params else evaluate_params[
        'params']
    returns_name = 'a_ret' if 'a_ret' not in evaluate_params else evaluate_params[
        'returns_name']
    returns_series = df[returns_name]

    if 'rolling' in method:
        fitness_series = empyrical.calculate(returns_series=returns_series,
                                             method=method,
                                             **params)
    elif 'series' in method:
        fitness_series = time_series(returns_series=returns_series, **params)
    else:
        fitness = empyrical.calculate(returns_series=returns_series,
                                      method=method,
                                      **params)

    if ('roll' in method
            or 'series' in method) and 'callback_series' in evaluate_params:
        fitness = evaluate_params['callback_series'](fitness_series)
    else:
        fitness = fitness_series.mean()

    return fitness
