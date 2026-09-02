import os
import pandas as pd
from pymongo import InsertOne, DeleteOne
from chaosmind.timing.sirius0003.workflow import WorkFlow
from kdutil.mongodb import MongoDBManager
from toolix.params import load_sirius_params
from toolix.macro.contract import *


class Signalor(object):

    def __init__(self, task_id, code, symbol):
        self.code = code
        self.task_id = task_id
        self.symbol = symbol
        self.max_window = 60
        self.init()

    def init(self):
        self._mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
        factors_infos, params = load_sirius_params(code=self.code,
                                                   task_id=self.task_id)
        self.workflow = WorkFlow(
            directory=params['model_path'],
            code=self.code,
            symbol=self.symbol,
            task_id=str(self.task_id),
            factors_infos=factors_infos,
            softmax_temperature=params['softmax_temperature'],
            min_open_signal_abs=params['min_open_signal_abs'],
            method=params['method'],
            win=params['win'],
            period=params['horizon'],
            signal_method=params['signal_method'],
            signal_params=params['signal_params'])

        self.workflow.initialization(mongo_client=self._mongo_client)

    def fetch_impluse(self, max_window, end_time, features):
        count = len(features)
        rt = self._mongo_client['neutron'][RAW_FACTORS_TABLE].find({
            'symbol':
            self.symbol,
            "name": {
                "$in": features
            },
            "datetime": {
                "$lte": end_time#.strftime('%Y-%m-%d %H:%M:%S')
            }
        }).sort([("datetime", -1)]).limit(max_window * count)
        
        impluse_data = pd.DataFrame(rt)
        impluse_data = impluse_data.sort_values(by='datetime')
        impluse_data = impluse_data.set_index(
            ['datetime', 'symbol',
             'name'])['value'].unstack().unstack().fillna(
                 method='ffill').fillna(0).stack()
        impluse_data = impluse_data.reset_index()
        return impluse_data

    def update_data(self, data, table_name):
        insert_request = [
            InsertOne(data) for data in data.to_dict(orient='records')
        ]
        delete_request = [
            DeleteOne(data)
            for data in data[['trade_time', 'code', 'symbol', 'task_id'
                              ]].to_dict(orient='records')
        ]
        _ = self._mongo_client['neutron'][table_name].bulk_write(
            delete_request + insert_request, bypass_document_validation=True)
        
        
    def wrap(self, data, trade_time, task_id, code, method=None, win=None):
        data1 = data.loc[[trade_time]].reset_index()
        data1 = data1.melt(id_vars=['trade_time', 'code'],
                           var_name='name',
                           value_name='value')
        data1['task_id'] = task_id
        data1['symbol'] = data1['code']
        data1['code'] = code
        if isinstance(method, str):
            data1['method'] = method
        if isinstance(win, int) or isinstance(win, str):
            data1['win'] = win
        return data1

    def run(self, trade_time):

        # temp_keys = {
        #     'tc004_2_2_3_0': 'tc004_1_1_2_0',
        #     'tc022_2_3_1': 'tv004_1_2_0',
        #     'tc022_5_10_1': 'tv007_1_2_1'
        # }
        #features = list(self.workflow.dependencies + list(temp_keys.keys()))
        features = self.workflow.dependencies 
        impluse_data = self.fetch_impluse(features=features,
                                          max_window=self.workflow.win * 4,
                                          end_time=trade_time)

        impluse_data = impluse_data.rename(columns={
            'datetime': 'trade_time',
            'symbol': 'code'
        })
        impluse_data['trade_time'] = pd.to_datetime(impluse_data['trade_time'])

        # impluse_data = impluse_data.rename(columns=temp_keys)

        if impluse_data[impluse_data['trade_time'] == trade_time].empty:
            return

        original_factors, normal_factors = self.workflow.create_factors(
            total_data=impluse_data)

        
        net_er_out = self.workflow.create_values(trade_time=trade_time,
                                                 data=normal_factors)
        events = self.workflow.conversion_signals(trade_time=trade_time,
                                                  raw_action=pd.DataFrame(
                                                      [net_er_out]),
                                                  name='net_er_out')
        
        original_factors = self.wrap(data=original_factors,
                                     trade_time=trade_time,
                                     code=self.code,
                                     task_id=self.workflow.task_id)

        normal_factors = self.wrap(data=normal_factors,
                                   trade_time=trade_time,
                                   code=self.code,
                                   task_id=self.workflow.task_id,
                                   method=self.workflow.method,
                                   win=self.workflow.win)
        ## 存储特征和存储信号
        if len(events) > 0:
            events = pd.DataFrame(events)
            events['symbol'] = self.symbol
            events['task_id'] = self.task_id
            self.update_data(data=pd.DataFrame(events),
                         table_name=TRADER_EVENT_TABLE)

        self.update_data(data=pd.DataFrame([net_er_out]),
                         table_name=TRADER_BIAS_TABLE)

        self.update_data(data=original_factors, table_name=DERIV_FACTORS_TABLE)

        self.update_data(data=normal_factors, table_name=NORM_FACTORS_TABLE)
