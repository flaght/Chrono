import numpy as np

def create_chg(market_data, name='vwap'):
    pricep = market_data.set_index(['trade_time', 'code'])[name].unstack()
    pre_pricep = pricep.shift(1)
    ret_v2v = np.log((pricep) / pre_pricep)
    yields_data = ret_v2v.shift(-2)
    yields_data = yields_data.stack()
    yields_data.name = 'chg_pct'
    return yields_data.reset_index()

def create_yields(data, horizon, offset=0):
    df = data.copy()
    df.set_index("trade_time", inplace=True)
    ## chg为log收益
    df['nxt1_ret'] = df['chg_pct']
    df = df.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    df = df.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    df.name = 'nxt1_ret'
    return df