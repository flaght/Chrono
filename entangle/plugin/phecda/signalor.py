import os, pdb
import pandas as pd
from pymongo import InsertOne, DeleteOne
from kdutil.mongodb import MongoDBManager
from chaosmind.timing.phecda0002.workflow import WorkFlow
from toolix.macro.contract import MAIN_CONTRACT_MAPPING, CONT_MULTNUM_MAPPING


class Signalor(object):

    def __init__(self, id, code):
        self.code = code
        self.id = id
        self.symbol = MAIN_CONTRACT_MAPPING[code]
        self.workflow = WorkFlow(
            code=code,
            symbol=MAIN_CONTRACT_MAPPING[code],
            base_path=os.environ['CHAOS_PHECDA_PATH'] ,
            name=id,
        )
        self._mongo_client = MongoDBManager(uri=os.environ['MG_URI'])

    def fetch_bar(self, begin_time, end_time):
        rt = self._mongo_client['neutron']['market_bar'].find({
            'symbol': self.symbol,
            "datetime": {
                "$lte": end_time.strftime('%Y-%m-%d %H:%M:%S'),
                "$gte": begin_time.strftime('%Y-%m-%d %H:%M:%S')
            }
        })
        data = pd.DataFrame(rt)
        return data.sort_values(by='datetime')

    def fetch_factors(self, trade_time, pos):
        ## 读取因子
        rt = self._mongo_client['neutron']['impluse_factors'].find(
            {
                'symbol': {
                    "$in": [self.symbol]
                },
                "datetime": {
                    "$lte": trade_time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }, {
                'datetime': 1,
                'symbol': 1,
                'name': 1,
                'value': 1
            }).sort([("datetime", -1)]).limit(pos * 1000)
        data = pd.DataFrame(rt)
        if data.empty:
            return data
        data.rename(columns={
            'datetime': 'trade_time',
            'symbol': 'code'
        },
                    inplace=True)
        data['trade_time'] = pd.to_datetime(data['trade_time'])
        data = data.set_index(
            ['trade_time', 'code',
             'name'])['value'].unstack().fillna(method='ffill').reset_index()
        return data

    def update_signalor(self, data, table_name):
        insert_request = [
            InsertOne(data) for data in data.to_dict(orient='records')
        ]

        delete_request = [
            DeleteOne(data)
            for data in data[['trade_time', 'symbol', 'task_id']].to_dict(
                orient='records')
        ]
        _ = self._mongo_client['neutron'][table_name].bulk_write(
            delete_request + insert_request, bypass_document_validation=True)

    def run(self, trade_time):
        ## 读取因子
        impluse_data = self.fetch_factors(trade_time=trade_time, pos=120)
        if impluse_data.empty:
            return
        min_time = impluse_data['trade_time'].min()
        max_time = impluse_data['trade_time'].max()
        bar_data = self.fetch_bar(begin_time=min_time, end_time=max_time)
        bar_data = bar_data[[
            'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'open_interest'
        ]]
        bar_data.rename(columns={
            'open_interest': 'openint',
            'datetime': 'trade_time',
            'symbol': 'code'
        },
                        inplace=True)
        bar_data['trade_time'] = pd.to_datetime(bar_data['trade_time'])
        impluse_data = impluse_data.merge(bar_data, on=['trade_time', 'code'])
        signal = self.workflow.create_signals(trade_time=trade_time,
                                                data=impluse_data)
        #signal = signal if signal == 0 else 0 - signal
        results = {
            'trade_time': trade_time,
            'signal': signal,
            'symbol': self.symbol,
            'task_id': self.id
        }
        self.update_signalor(data=pd.DataFrame([results]),
                             table_name='impluse_signal')
