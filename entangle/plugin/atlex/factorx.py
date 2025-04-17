import os, pdb
import pandas as pd
from pymongo import InsertOne, DeleteOne
from kdutil.mongodb import MongoDBManager
from lumina.formual.iactuator import Iactuator


class Factorx(object):

    def __init__(self, symbol, n_job=1):
        self.symbol = symbol
        self._mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
        self._iactuator = Iactuator(k_split=n_job)

    def fetch_bar(self, trade_time, pos):
        rt = self._mongo_client['neutron']['market_bar'].find({
            'symbol': self.symbol,
            "datetime": {
                "$lte": trade_time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }).sort([("datetime", -1)]).limit(pos)
        data = pd.DataFrame(rt)
        return data.sort_values(by='datetime')

    def impluse_run(self, trade_time):

        def _format(data, impluse_max):
            data.index.names = ['datetime', 'symbol', 'name']
            data.name = 'value'
            data = data.reset_index()
            data.index = data['datetime'].factorize()[0]
            data = data.loc[impluse_max - 1:]
            return data.reset_index(drop=True)

        impluse_max = 120#self.formual_client.impulse.max_window()
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
        bar_data = bar_data.set_index(['datetime', 'symbol'])
        res = {}
        for col in cols:
            if col not in bar_data.columns:
                continue
            res[col] = bar_data[col].unstack()
        #impluse_data = self.formual_client.impulse.batch(data=res)
        impluse_data = self._iactuator.calculate(total_data=res)
        impluse_data = _format(impluse_data.stack(), impluse_max)
        self.update_impluse(data=impluse_data, table_name='impluse_factors')
        return impluse_data
