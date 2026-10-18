##处理DX的 level2数据
import pdb, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from alphacopilot.api.data import RetrievalAPI
from kdutils.macro2 import *
from kdutils.ttimes import get_dates
from kdutils.tactix import Tactix


def save(factors_data, code, method, instruments, name):
    dirs = os.path.join(base_path, method, instruments, 'level2')
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    fd = factors_data[factors_data.code.isin([code])]
    pdb.set_trace()
    filename = os.path.join(dirs, f"{name}_data.feather")
    print(filename)
    fd.reset_index(drop=True).to_feather(filename)


def start(method, filename):
    pdb.set_trace()
    start_date, end_date = get_dates(method)
    filename = os.path.join(filename)
    factors_data = pd.read_parquet(filename)
    #factors_data = factors_data.reset_index()
    factors_data = factors_data.rename(columns={'Code': 'symbol'})
    factors_data['minTime'] = factors_data['minTime'].astype(str).str.zfill(6)
    datetime_str = factors_data['date'].astype(
        str) + factors_data['minTime'].astype(str)
    factors_data['trade_time'] = pd.to_datetime(datetime_str,
                                                format='%Y%m%d%H%M%S')
    factors_data = factors_data.drop(columns=['date', 'minTime'])
    regex_pattern = r'^([A-Za-z]+)'
    factors_data['code'] = factors_data['symbol'].str.extract(regex_pattern)
    factors_data = factors_data.set_index(
        'trade_time').loc[start_date:end_date].reset_index()
    
    factors_data['trade_time'] = pd.to_datetime(
        factors_data['trade_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    pdb.set_trace()
    algin_data = RetrievalAPI.get_algin_factors(
        begin_date=start_date, end_date=end_date, 
        codes=factors_data['code'].unique().tolist())
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    factors_data['trade_date'] = factors_data['trade_time'].dt.normalize()
    algin_data['trade_date'] = pd.to_datetime(algin_data['trade_date'])
    dominant_contracts_lookup = algin_data[['trade_date', 'symbol']].drop_duplicates().copy()
    merged_data = pd.merge(factors_data,dominant_contracts_lookup,on=['trade_date', 'symbol'],how='inner')
    factors_data = merged_data.copy()
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    codes = factors_data.code.unique().tolist()
    mapping = {'IM': 'ims', 'IC': 'ics', 'IF': 'ifs', 'IH': 'ihs'}
    for code in codes:
        instruments = mapping[code]
        total_data = factors_data.copy()
        total_data = total_data[total_data.code.isin([code])]
        times = total_data['trade_time'].unique().tolist()
        len1 = round(len(times) * 0.6)  # 70%部分
        len2 = round(len(times) * 0.2)  # 20%部分
        len3 = len(times) - len1 - len2
        pdb.set_trace()
        train_data = total_data[total_data['trade_time'].isin(times[:len1])]
        val_data = total_data[total_data['trade_time'].isin(times[len1:len1 +
                                                                  len2])]
        test_data = total_data[total_data['trade_time'].isin(times[len1 +
                                                                   len2:])]
        pdb.set_trace()
        save(factors_data=train_data[train_data.code.isin([code])],
             code=code,
             method=method,
             instruments=instruments,
             name='train')
        save(factors_data=val_data[val_data.code.isin([code])],
             code=code,
             method=method,
             instruments=instruments,
             name='val')
        save(factors_data=test_data[test_data.code.isin([code])],
             code=code,
             method=method,
             instruments=instruments,
             name='test')

if __name__ == '__main__':
    variant = Tactix().start()
    start(method=variant.method,
          filename=variant.filename)

#run(method='cicso0')
