##处理BN的 KLINE数据
import pdb, os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from kdutils.macro2 import *
from kdutils.ttimes import get_dates
from kdutils.tactix import Tactix

def save(factors_data, method, instruments, name):
    pdb.set_trace()
    dirs = os.path.join(base_path, method, instruments, 'basic')
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    fd = factors_data
    filename = os.path.join(dirs, f"{name}_data.feather")
    print(filename)
    fd.reset_index(drop=True).to_feather(filename)



def start(method, category, freq, instruments, types='futures'):
    start_date, end_date = get_dates(method)
    dirs = os.path.join(data_path, types, category, 'klines', freq, INSTRUMENTS_CODES[instruments])
    rootp = Path(dirs)
    file_list = (p for p in rootp.rglob('*.csv') if p.is_file())
    file_list = [p for p in file_list]
    pdb.set_trace()

    res = []
    for p in file_list:
        print(p)
        data = pd.read_csv(p)
        data = data.rename(columns={'quote_volume':'value','count':'deal'})
        data = data.drop(['ignore','open_time','close_time','taker_buy_volume','taker_buy_quote_volume'],axis=1)
        data['vwap'] = data['value'] / data['volume']
        data['code'] = INSTRUMENTS_CODES[instruments]
        res.append(data)
    factors_data = pd.concat(res,axis=0)
    pdb.set_trace()
    factors_data = factors_data.sort_values(by=['trade_time','code'])
    factors_data = factors_data.set_index('trade_time').loc[start_date:end_date].reset_index()
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    total_data = factors_data.copy()
    times = total_data['trade_time'].unique().tolist()
    len1 = round(len(times) * 0.6)  # 60%部分
    len2 = round(len(times) * 0.2)  # 20%部分
    len3 = len(times) - len1 - len2
    train_data = total_data[total_data['trade_time'].isin(times[:len1])]
    val_data = total_data[total_data['trade_time'].isin(times[len1:len1 +
                                                                  len2])]
    test_data = total_data[total_data['trade_time'].isin(times[len1 +
                                                                   len2:])]

    train_data['trade_time'] = pd.to_datetime(train_data['trade_time'])
    val_data['trade_time'] = pd.to_datetime(val_data['trade_time'])
    test_data['trade_time'] = pd.to_datetime(test_data['trade_time'])
    total_data['trade_time'] = pd.to_datetime(total_data['trade_time'])

    save(factors_data=train_data,method=method,
            instruments=instruments, name='train')

    save(factors_data=val_data,method=method,
            instruments=instruments, name='val')

    save(factors_data=test_data,method=method,
            instruments=instruments, name='test')
    
    save(factors_data=total_data,method=method,
            instruments=instruments, name='all')



if __name__ == '__main__':
    variant = Tactix().start()
    start(method=variant.method,
          category=variant.category,
          freq=variant.freq,
          instruments=variant.instruments)