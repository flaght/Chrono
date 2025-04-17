import os, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from alphacopilot.api.data import RetrievalAPI


def get_dates(method):
    if method == 'train':
        return '2018-01-01', '2023-01-01'
    elif method == 'val':
        return '2023-01-01', '2024-01-01'
    elif method == 'test':
        return '2024-01-01', '2024-07-01'
    elif method == 'micro':
        return '2021-01-01', '2023-01-01'
    elif method == 'min_train':
        return '2023-01-01', '2023-07-01'
    elif method == 'min_val':
        return '2023-07-01', '2024-01-01'
    elif method == 'min_evolution':
        return '2023-01-01', '2024-01-01'


def fetch_data(method):
    codes = ['IM', 'IH', 'IC', 'IF']
    begin_date, end_date = get_dates(method=method)
    data = RetrievalAPI.get_main_price(begin_date=begin_date,
                                       end_date=end_date,
                                       codes=codes,
                                       method='pcr',
                                       format_data=0)

    for code in data.keys():
        dt = data[code]
        dt['trade_time'] = pd.to_datetime(dt['barTime'])
        dt.rename(columns={
            'closePrice': 'close',
            'lowPrice': 'low',
            'highPrice': 'high',
            'openPrice': 'open',
            'totalVolume': 'volume',
            'totalValue': 'value',
            'openInterest': 'openint',
            'logRet': 'chg'
        },
                  inplace=True)
        dt = dt.drop(columns=['barTime', 'symbol', 'mincount', 'trade_date'],
                     axis=1)
        dt['price'] = dt['value'] / dt[
            'volume']  #此处用于成交价，但会出现value volume为0情况，导致price为inf，此情况使用 olch均值代替
        dt['price'] = dt['price'].where(
            dt['price'].notna(), dt[['high', 'low', 'close',
                                     'open']].mean(axis=1))
        file_path = os.path.join(os.getenv('BASE_PATH'), 'times', code)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        filename = os.path.join(file_path, f'{method}.feather')
        print(f'Saving {filename}')
        dt.reset_index(drop=True).to_feather(filename)


def main(method):
    fetch_data(method)


main(os.getenv('METHOD'))
