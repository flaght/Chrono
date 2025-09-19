### 整理之前的因子，绩效入库
import pdb, os, argparse, itertools
import pandas as pd
import numpy as np
from pymongo import InsertOne, DeleteOne
from dotenv import load_dotenv

load_dotenv()
from ultron.factor.genetic.geneticist.operators import calc_factor
from ultron.sentry.api import *
from kdutils.mongodb import MongoDBManager
from kdutils.common import fetch_temp_data
from lumina.genetic.metrics.evaluate import FactorEvaluate
from lumina.genetic.process import *


@add_process_env_sig
def callback_fitness(target_column, method, instruments, source, fee,
                     total_data, returns_data):
    res = []
    for column in target_column:
        print(column)
        try:
            results = calc_fitness(target_column=column,
                                   method=method,
                                   instruments=instruments,
                                   fee=fee,
                                   source=source,
                                   total_data=total_data,
                                   returns_data=returns_data)
            res.append(results)
        except Exception as e:
            print("error:{0}:{1}".format(column, str(e)))
    return res


def calc_fitness(target_column, method, instruments, source, fee, total_data,
                 returns_data):
    expression = target_column['expression']
    params = target_column['params']
    #scale_method = target_column['scale_method']
    #roll_win = target_column['roll_win']
    #ret_name = target_column['ret_name']
    factor_data = calc_factor(expression=expression,
                              total_data=total_data,
                              key='code',
                              indexs=[])
    factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
    factor_data['transformed'] = np.where(
        np.abs(factor_data.transformed.values) > 0.000001,
        factor_data.transformed.values, np.nan)
    factor_data = factor_data.loc[factor_data.index.unique()[1:]]
    factors_data1 = factor_data.reset_index()
    total_data1 = factors_data1.merge(returns_data, on=['trade_time',
                                                        'code']).dropna()
    res = []
    for param in params:
        scale_method = param['scale_method']
        roll_win = param['roll_win']
        ret_name = param['ret_name']
        MyFactorBacktest = FactorEvaluate(
            factor_data=total_data1,
            factor_name='transformed',
            ret_name=ret_name,
            roll_win=roll_win,  # 因子放缩窗口，自定义
            fee=fee,
            scale_method=scale_method)  # 可换 'roll_zscore' 等

        result = MyFactorBacktest.run()
        result['name'] = target_column['name']
        result['scale_method'] = scale_method
        result['roll_win'] = roll_win
        result['ret_name'] = ret_name
        result['method'] = method
        result['instruments'] = instruments
        result['source'] = source
        '''
        result['nav'] = MyFactorBacktest.factor_data['nav'].reset_index()  ## 净值 累加
        result['gross_ret'] = MyFactorBacktest.factor_data[
            'gross_ret'].reset_index().to_dict(orient='records')  #收益（未扣除费用）

        result['net_ret'] = MyFactorBacktest.factor_data[
            'net_ret'].reset_index().to_dict(orient='records')  # 收益（扣除费用）

        result['turnover'] = MyFactorBacktest.factor_data[
            'turnover'].reset_index().to_dict(orient='records')  ## 换手率
        '''
        res.append(pd.DataFrame([result]))
    return res


class MergeFactors(object):

    def __init__(self):
        self._mongo_client = MongoDBManager(uri=os.environ['MG_URI'])

    ## 文件读取 若需要刷新 则从另外函数执行
    def fetch_market(self, instruments, method, task_id, name):
        total_data = fetch_temp_data(
            method=method,
            task_id=task_id,
            instruments=instruments,
            datasets=name if isinstance(name, list) else [name])

        total_returns = fetch_temp_data(
            method=method,
            task_id=task_id,
            instruments=instruments,
            datasets=name if isinstance(name, list) else [name],
            category='returns')
        return total_data, total_returns

    def fetch_expression_mongo(self, trade_time, score=4, category="basic"):
        results = self._mongo_client['neutron']['quvse_factors_details'].find(
            {
                'score': {
                    "$gte": score
                },
                "timestampe": {
                    "$gte": trade_time
                },
                "category": category
            }, {
                'expression': 1,
                'score': 1,
                'name': 1
            })

        data = pd.DataFrame(results)
        return data

    def fetch_expression_file(self, method, task_id):
        temp1 = os.path.join('temp', method, '200036', 'evolution',
                             'programs_{0}.feather'.format('200036'))
        ms1 = pd.read_feather(temp1).drop(['update_time'],
                                          axis=1)[['name', 'formual']]
        return ms1.rename(columns={'formual': 'expression'})

    def update_results(self, table_name, data, keys=[]):
        insert_request = [
            InsertOne(data) for data in data.to_dict(orient='records')
        ]

        delete_request = [
            DeleteOne(data) for data in data[keys].to_dict(orient='records')
        ]
        _ = self._mongo_client['neutron'][table_name].bulk_write(
            delete_request + insert_request, bypass_document_validation=True)

    def update_data(self, results):
        res1 = list(itertools.chain.from_iterable(results))
        res1 = [rs for res in res1 for rs in res]
        result = pd.concat(res1)
        ## 保留正常的
        result = result.dropna(subset=['calmar', 'ic_mean', 'ic_std', 'ic_ir'])

        result = result[result.calmar < 10]
        self.update_results(table_name='abily_temp_factemp',
                            data=result,
                            keys=[
                                'name', 'scale_method', 'roll_win', 'ret_name',
                                'method', 'instruments'
                            ])

    def calc(self,
             expression_data,
             scale_method_sets,
             roll_win_sets,
             returns_columns,
             method,
             instruments,
             source,
             total_data,
             returns_data,
             k_split=1,
             num=20):

        params_sets = [
            {
                'scale_method': scale_method,
                'roll_win': roll_win,
                'ret_name': return_col
            } for scale_method in scale_method_sets  # 遍历每个放缩方法
            for roll_win in roll_win_sets  # 遍历每个滚动窗口
            for return_col in returns_columns
        ]
        res = [
            {
                **expr_dict_original,
                'params': params_sets,
            } for expr_dict_original in expression_data  # 遍历每个原始表达式字典
        ]

        process_list = split_k(k_split, res[:])
        res1 = create_parellel(process_list=process_list,
                               callback=callback_fitness,
                               method=method,
                               fee=0.0000005,
                               instruments=instruments,
                               source=source,
                               total_data=total_data,
                               returns_data=returns_data)

        self.update_data(results=res1)

    def run(self, method, instruments, task_id, source):

        def split_list(l, n):
            return [l[x:x + n] for x in range(0, len(l), n)]

        total_data, total_returns = self.fetch_market(
            instruments=instruments,
            method=method,
            task_id=task_id,
            name=['train', 'val', 'test'])
        total_data = total_data.sort_values(by=['trade_time', 'code'])
        total_returns = total_returns.sort_values(by=['trade_time', 'code'])
        total_data = total_data.set_index('trade_time')

        if source == 'llm':
            expression_data = self.fetch_expression_mongo(
                trade_time='2025-09-01', score=4)
        elif source == 'evo':
            expression_data = self.fetch_expression_file(method=method,
                                                         task_id=task_id)
        expression_data = expression_data.to_dict(orient='records')
        expression_data_sets = split_list(expression_data, 20)

        scale_method_sets = [
            'roll_min_max',
            'roll_zscore',
            'roll_quantile',
            'ew_zscore'  #,'train_const'
        ]
        returns_columns = ['time_weight', 'equal_weight']
        roll_win_sets = [60, 120, 240, 300]
        ## 分页计算
        pdb.set_trace()
        for ed in expression_data_sets:
            results = self.calc(expression_data=ed,
                            scale_method_sets=scale_method_sets,
                            roll_win_sets=roll_win_sets,
                            returns_columns=returns_columns,
                            method=method,
                            instruments=instruments,
                            source=source,
                            total_data=total_data,
                            returns_data=total_returns,
                            k_split=4)
        pdb.set_trace()
        print('-->')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--method',
                        type=str,
                        default='aicso0',
                        help='data method')
    parser.add_argument('--instruments',
                        type=str,
                        default='ics',
                        help='code or instruments')

    parser.add_argument('--task_id',
                        type=str,
                        default='140001',
                        help='code or instruments')

    parser.add_argument('--source',
                        type=str,
                        default='evo',
                        help='code or instruments')

    args = parser.parse_args()
    MergeFactors().run(method=args.method,
                       instruments=args.instruments,
                       source=args.source,
                       task_id=args.task_id)
