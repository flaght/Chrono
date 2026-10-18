import os, pdb
import warnings
import pandas as pd
from pymongo import InsertOne, DeleteOne
from kdutil.mongodb import MongoDBManager
from lumina.formual.iactuator import Iactuator
from toolix.macro.contract import *


class Factorx(object):

    def __init__(self, symbol, n_job=1, impulse=[]):
        self.symbol = symbol
        ## 复权因子
        self.adjusted_factor = ADUJSTED_FACTOR_MAPPING[symbol]
        self._mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
        self._iactuator = Iactuator(k_split=n_job, impulse=impulse)

    def fetch_bar(self, trade_time, pos):
        rt = self._mongo_client['neutron'][MARKET_BAR_TABLE].find({
            'symbol':
            self.symbol,
            "datetime": {
                "$lte": trade_time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }).sort([("datetime", -1)]).limit(pos)
        data = pd.DataFrame(rt)
        ## 复权计算
        data[[
            'open', 'high', 'low', 'close'
        ]] = data[['open', 'high', 'low', 'close']] * self.adjusted_factor
        data['datetime'] = pd.to_datetime(data['datetime'])
        return data.sort_values(by='datetime')

    def update_impluse(self, data, table_name):
        insert_request = [
            InsertOne(data) for data in data.to_dict(orient='records')
        ]

        delete_request = [
            DeleteOne(data)
            for data in data[['datetime', 'symbol', 'name']].to_dict(
                orient='records')
        ]
        _ = self._mongo_client['neutron'][table_name].bulk_write(
            delete_request + insert_request, bypass_document_validation=True)

    def impluse_run(self, trade_time):

        def _format(data, impluse_max):
            data.index.names = ['datetime', 'symbol', 'name']
            data.name = 'value'
            data = data.reset_index()
            data.index = data['datetime'].factorize()[0]
            data = data.loc[impluse_max - 1:]
            return data.reset_index(drop=True)


        impluse_max = 120  #self.formual_client.impulse.max_window()
        
        bar_data = self.fetch_bar(trade_time=trade_time, pos=impluse_max)
    
        if bar_data.shape[0] < impluse_max:
            print(
                f'Not enough data for {impluse_max} window {self.symbol} at {trade_time}'
            )
            return pd.DataFrame()
        bar_data.rename(columns={'open_interest': 'openint'}, inplace=True)
        cols = [
            'open', 'high', 'low', 'close', 'volume', 'value', 'openint',
            'vwap'
        ]
        
        if bar_data[bar_data['datetime'] == trade_time].empty:
            print("{} not bar data".format(trade_time))
            return pd.DataFrame()
        bar_data = bar_data.set_index(['datetime', 'symbol'])
        res = {}
        for col in cols:
            if col not in bar_data.columns:
                continue
            res[col] = bar_data[col].unstack().fillna(method='ffill')
            
        impluse_data = self._iactuator.calculate(total_data=res)
        impluse_data = _format(impluse_data.stack(), impluse_max)
        impluse_data = impluse_data.drop_duplicates(subset=['datetime','symbol','name'])
        self.update_impluse(data=impluse_data, table_name=RAW_FACTORS_TABLE)
        return impluse_data
