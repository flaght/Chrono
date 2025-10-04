import pdb
import numpy as np
from lib.aux001 import fetch_market

aggregation_rules = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'value': 'sum',
    'volume': 'sum',  # 成交量是流量，用 'sum'
    'openint': 'last'  # 持仓量是存量，用 'last'
}


def fetch_data(method,
               instruments,
               task_id,
               datasets=['train', 'val', 'test']):
    factors_data, returns_data = fetch_market(instruments=instruments,
                                              method=method,
                                              task_id=task_id,
                                              datasets=datasets)
    total_data = factors_data.merge(returns_data, on=['trade_time', 'code'])
    '''
    nxt1_columns = total_data.filter(regex="^nxt1").columns.to_list()
    
    basic_columns = [
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint'
    ]
    factor_columns = [
        col for col in total_data.columns
        if col not in ['trade_time', 'code'] + nxt1_columns + basic_columns
    ]
    '''
    #return_name = "nxt1_ret_{}h".format(period)

    #total_data.rename(columns={return_name: 'nxt1_ret'}, inplace=True)
    return total_data.sort_values(by=['trade_time', 'code'])


def aggregation_data(factor_data, returns_data, period):
    dt = factor_data.merge(
        returns_data[['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)]],
        on=['trade_time', 'code'])
    is_on_mark = dt['trade_time'].dt.minute % int(period) == 0
    dt = dt[is_on_mark]
    dt.replace([np.inf, -np.inf], np.nan, inplace=True)
    dt.dropna(inplace=True)
    return dt


def fetch_times(method, task_id, instruments):
    train_data = fetch_data(method=method,
                            task_id=task_id,
                            instruments=instruments,
                            datasets=['train'])
    val_data = fetch_data(method=method,
                          task_id=task_id,
                          instruments=instruments,
                          datasets=['val'])
    test_data = fetch_data(method=method,
                           task_id=task_id,
                           instruments=instruments,
                           datasets=['test'])
    return {
        'train_time':
        (train_data['trade_time'].min(), train_data['trade_time'].max()),
        'val_time':
        (val_data['trade_time'].min(), val_data['trade_time'].max()),
        'test_time':
        (test_data['trade_time'].min(), test_data['trade_time'].max())
    }
