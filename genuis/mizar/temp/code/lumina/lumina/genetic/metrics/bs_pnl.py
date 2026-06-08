import numpy as np
import pandas as pd


def calculate_pnl(pos_data,
                  total_data,
                  strategy_settings,
                  is_trade_vol=True) -> pd.DataFrame:

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
    commission_rate = strategy_settings.get('commission',
                                            0.0)  # commission_rate=10/10000
    if commission_rate > 0.1:
        commission_rate = commission_rate / 10000

    print("手续费:", round(commission_rate * 10000, 2), "%%")
    slippage_point = np.array(strategy_settings.get('slippage', 0.0))
    size = np.array(strategy_settings.get('size', 10))

    codes = pos_data.columns.get_level_values(1).unique().tolist()
    size_list = [strategy_settings.get('size', 10)[code] for code in codes]
    size = np.array(size_list).reshape(1, -1)

    df0t = total_data.loc[:, ('close', codes)]

    df0 = df0t.join(pos_data, how='left').fillna(0)

    close = df0['close']

    if is_trade_vol:
        trade_vol = total_data.loc[:, ('trade_vol', codes)]
        trade_vol = trade_vol.join(pos_data, how='left').fillna(0)
        pos = df0['pos'] * trade_vol['trade_vol']
    else:
        pos = df0['pos']

    pos = pos.fillna(0)
    df2 = total_data.loc[:, ('open', codes)]
    openD = df2['open']
    change = (openD.diff(1)).shift(-1).fillna(0)  # 修改
    # 生成交易记录

    trade = (pos.diff()).fillna(0)

    commission = abs(trade) * openD * commission_rate * size  # 修改

    # commission.sum()

    slippage = abs(trade) * slippage_point * size
    # 处理成开盘价成交模式 1
    # 开仓的时候

    #change2=openD-close.shift(1)
    #openD2=openD
    #yk=change * pos * size
    pnl = change * pos * size - commission - slippage

    # 处理成开盘价成交模式 2
    #pnl = change * pos * size - commission - slippage
    ret = pnl / capital

    balance = len(codes) * capital + pnl.sum(1).cumsum()
    balance = balance.to_frame('balance')

    pnl2 = pnl.sum(1).to_frame('pnl')
    ret = ret.sum(1).to_frame('ret')
    drawdown = (balance -
                balance.cummax()).rename(columns={'balance': 'drawdown'})

    close['close'] = close[codes[0]]
    closeD = close['close']
    df = pd.concat([balance, drawdown, pnl2, ret, pnl, closeD], axis=1)
    ddD = {
        'balance': 'last',
        'drawdown': 'last',
        'pnl': 'sum',
        'ret': 'sum',
        'close': 'last',
    }

    for code in codes:
        ddD[code] = 'sum'

    df = df.resample('1D').agg(ddD).dropna().fillna({
        'pnl': 0,
        'ret': 0,
    }).fillna(method='ffill')
    # df['balance'].values[-1]
    print('总盈利=', round(df['pnl'].sum() / 10000, 2), "万")

    df['ret'] = df['pnl'] / capital / len(codes) * 2

    return df
