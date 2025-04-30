import os, pdb, pickle
import pandas as pd
import numpy as np
from lumina.formual.base import FormualBase
from pymongo import InsertOne, DeleteOne
from kdutil.mongodb import MongoDBManager
from toolix.macro.contract import MAIN_CONTRACT_MAPPING
from chaosmind.timing.virgo0001.workflow import WorkFlow


class Signalor(object):

    def __init__(self, id, code, symbol):
        self.code = code
        self.id = id
        self.symbol = symbol
        self._mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
        self.init_params(n_job=4)

    def _create_data(self, end_time, max_window, features):
        end_timestamp = pd.to_datetime(end_time)
        date_range = pd.date_range(end=end_timestamp,
                                   periods=max_window,
                                   freq='T')
        df = pd.DataFrame({
            'trade_time': date_range,
            **{
                feature: np.random.rand(max_window)
                for feature in features
            }  # 生成0-1之间的随机数
        })
        df['code'] = self.code
        df['price'] = 0
        return df

    def init_params(self, n_job):
        filename = os.path.join(os.environ['CHAOS_VIRGTOR_PATH'], self.code,
                                'param', "{0}.pkl".format(self.id))
        model_path = os.path.join(os.environ['CHAOS_VIRGTOR_PATH'], self.code,
                                  'model')
        params = pickle.load(open(filename, 'rb'))
        params['rewards_window'] = 10
        formual_data = pd.DataFrame(params['formual'])
        self.formual_client = FormualBase(formual=formual_data,
                                          task_id=params['task_id'],
                                          n_job=n_job)
        self.max_window = int(self.formual_client.impulse.max_window() * 1.5)
        ## 用于模型中初始化数据校验
        base_data = self._create_data(features=params['features'] + ['price'],
                                      max_window=5,
                                      end_time='2025-04-28 09:00:00')
        pdb.set_trace()
        self.workflow = WorkFlow(directory=model_path,
                                 code=self.code,
                                 symbol=self.symbol,
                                 log_path=os.environ['CHAOS_LOG_PATH'],
                                 base_data=base_data,
                                 model_name='sac_base',
                                 **params)

    def fetch_impluse(self, features, max_window, end_time):
        rt = self._mongo_client['neutron']['impluse_factors'].find({
            'symbol':
            self.symbol,
            "name": {
                "$in": features
            },
            "datetime": {
                "$lte": end_time
            }
        }).sort([("datetime", -1)]).limit(max_window * len(features))
        impluse_data = pd.DataFrame(rt)
        impluse_data = impluse_data.sort_values(by='datetime')
        impluse_data = impluse_data.set_index(
            ['datetime', 'symbol',
             'name'])['value'].unstack().unstack().fillna(
                 method='ffill').fillna(0).stack()
        impluse_data = impluse_data.reset_index()
        return impluse_data

    def fetch_bar(self, trade_time, symbol, pos):
        rt = self._mongo_client['neutron']['market_bar'].find({
            'symbol': symbol,
            "datetime": {
                "$lte": trade_time
            }
        }).sort([("datetime", -1)]).limit(pos)

        data = pd.DataFrame(rt)
        return data.sort_values(by='datetime')

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
        impluse_data = self.fetch_impluse(
            features=self.formual_client.dependencies,
            max_window=self.max_window,
            end_time=trade_time)
        bar_data = self.fetch_bar(trade_time=trade_time,
                                  symbol=self.symbol,
                                  pos=self.max_window)

        bar_data = bar_data[['datetime', 'symbol',
                             'close']].rename(columns={
                                 'datetime': 'trade_time',
                                 'symbol': 'code',
                                 'close': 'price'
                             })
        impluse_data = impluse_data.rename(columns={
            'datetime': 'trade_time',
            'symbol': 'code'
        })
        impluse_data['trade_time'] = pd.to_datetime(impluse_data['trade_time'])
        bar_data['trade_time'] = pd.to_datetime(bar_data['trade_time'])

        formual_data = self.formual_client.batch(data=impluse_data,
                                                 method='impulse1')

        formual_data = formual_data.set_index([
            'trade_time', 'code'
        ]).unstack().fillna(method='ffill').fillna(0).stack().reset_index()
        formual_data = formual_data.merge(bar_data, on=['trade_time', 'code'])
        signal = self.workflow.create_signals(trade_time=trade_time,
                                              data=formual_data)
        print(signal)
        #self.update_signalor(data=pd.DataFrame([signal]),
        #                     table_name='impluse_signals')
