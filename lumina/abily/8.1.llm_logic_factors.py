import pdb, os, argparse, itertools
import pandas as pd
import numpy as np
from ultron.tradingday import advanceDateByCalendar
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
from pymongo import InsertOne, DeleteOne

from ultron.factor.genetic.geneticist.operators import calc_factor
from ultron.sentry.api import *
from lumina.genetic.process import *
from lumina.genetic.metrics.evaluate import FactorEvaluate
from kdutils.macro2 import *

from kdutils.data import fetch_main_market
from kdutils.mongodb import MongoDBManager
#from nfactor import FactorEvaluate

mongo_client = MongoDBManager(uri=os.environ['MG_URI'])


@add_process_env_sig
def callback_fitness(target_column, method, instruments, fee, total_data,
                     returns_data):
    res = []
    for column in target_column:
        print(column)
        results = calc_fitness(column, method, instruments, fee, total_data,
                               returns_data)
        res.append(results)
    return res


def calc_fitness(target_column, method, instruments, fee, total_data,
                 returns_data):
    expression = target_column['expression']
    scale_method = target_column['scale_method']
    roll_win = target_column['roll_win']
    ret_name = target_column['ret_name']
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
    MyFactorBacktest = FactorEvaluate(
        factor_data=total_data1,
        factor_name='transformed',
        ret_name=ret_name,
        roll_win=roll_win,  # 因子放缩窗口，自定义
        fee=fee,
        scale_method=scale_method)  # 可换 'roll_zscore' 等

    results = MyFactorBacktest.run()
    results['name'] = target_column['name']
    results['scale_method'] = scale_method
    results['roll_win'] = roll_win
    results['ret_name'] = ret_name
    return results


def fetch_data(method, instruments):
    rootid = INDEX_MAPPING[INSTRUMENTS_CODES[instruments]]

    filename = os.path.join(
        base_path, method, instruments,
        DATAKIND_MAPPING[str(INDEX_MAPPING[INSTRUMENTS_CODES[instruments]])],
        'train_data.feather')
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    return factors_data


def create_chg(market_data, name='vwap'):
    pricep = market_data.set_index(['trade_time', 'code'])[name].unstack()
    pre_pricep = pricep.shift(1)
    ret_v2v = np.log((pricep) / pre_pricep)
    yields_data = ret_v2v.shift(-2)
    yields_data = yields_data.stack()
    yields_data.name = 'chg_pct'
    return yields_data.reset_index()


def create_yields(data, horizon, offset=0):
    df = data.copy()
    df.set_index("trade_time", inplace=True)
    ## chg为log收益
    df['nxt1_ret'] = df['chg_pct']
    df = df.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    df = df.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    df.name = 'nxt1_ret'
    return df


def fetch_returns(begin_date, end_date, codes):
    res = []
    horizon_sets = [15]
    market_data = fetch_main_market(begin_date=begin_date,
                                    end_date=end_date,
                                    codes=codes)
    chg_data = create_chg(market_data)
    for horizon in horizon_sets:
        df = create_yields(data=chg_data.copy(), horizon=horizon)
        df.name = "nxt1_ret_{0}h".format(horizon)
        res.append(df)
    data1 = pd.concat(res, axis=1)

    weights_raw = {
        'nxt1_ret_1h': 3,  # T+1 最大
        'nxt1_ret_2h': 2,  # T+2 其次
        'nxt1_ret_3h': 1  # T+3 最小
    }
    total_raw_weight = sum(weights_raw.values())
    weights = {col: w / total_raw_weight for col, w in weights_raw.items()}

    data1['time_weight'] = (data1['nxt1_ret_1h'] * weights['nxt1_ret_1h'] +
                            data1['nxt1_ret_2h'] * weights['nxt1_ret_2h'] +
                            data1['nxt1_ret_3h'] * weights['nxt1_ret_3h'])

    data1['equal_weight'] = data1[weights_raw.keys()].mean(axis=1)
    return data1


def prepare(method, instruments, task_id):
    factors_data = fetch_data(method, instruments)
    begin_date = advanceDateByCalendar("china.sse",
                                       factors_data['trade_time'].min(),
                                       '-5b').strftime('%Y-%m-%d')
    end_date = advanceDateByCalendar("china.sse",
                                     factors_data['trade_time'].max(),
                                     '-5b').strftime('%Y-%m-%d')
    returns_data = fetch_returns(begin_date=begin_date,
                                 end_date=end_date,
                                 codes=[INSTRUMENTS_CODES[instruments]])
    total_data = factors_data.merge(returns_data.reset_index(),
                                    on=['trade_time', 'code'])

    base_dirs = os.path.join(
        os.path.join('temp', "{}".format(method), task_id, 'llm'))
    if not os.path.exists(base_dirs):
        os.makedirs(base_dirs)
    pdb.set_trace()
    total_data.to_feather(os.path.join(base_dirs, "base.feather"))


def fetch_expression(trade_time, score):
    results = mongo_client['neutron']['quvse_factors_details'].find(
        {
            'score': {
                "$gte": score
            },
            "timestampe": {
                "$gte": trade_time
            }
        }, {
            'expression': 1,
            'score': 1,
            'name': 1
        })
    data = pd.DataFrame(results)
    return data


def update_results(data, keys=[]):
    insert_request = [
        InsertOne(data) for data in data.to_dict(orient='records')
    ]

    delete_request = [
        DeleteOne(data) for data in data[keys].to_dict(orient='records')
    ]
    _ = mongo_client['neutron']['abily_llm_factors'].bulk_write(
        delete_request + insert_request, bypass_document_validation=True)


def test1(method, instruments, task_id):
    k_split = 4
    filename = os.path.join(
        os.path.join('temp', "{}".format(method), task_id, 'llm',
                     'base.feather'))
    total_data = pd.read_feather(filename)
    total_data = total_data.sort_values(by=['trade_time', 'code'])

    returns_data = total_data[[
        'trade_time', 'code', 'nxt1_ret_1h', 'nxt1_ret_2h', 'nxt1_ret_3h',
        'time_weight', 'equal_weight'
    ]]
    total_data = total_data.set_index('trade_time')

    expression_data = fetch_expression(trade_time='2025-09-01', score=8)
    expression_data = expression_data.to_dict(orient='records')
    scale_method_sets = [
        'roll_min_max',
        'roll_zscore',
        'roll_quantile',
        'ew_zscore'  #,'train_const'
    ]
    returns_columns = ['time_weight', 'equal_weight']
    roll_win_sets = [60, 120, 240, 300]
    res = [
        {
            **expr_dict_original, 'scale_method': scale_method,
            'roll_win': roll_win,
            'ret_name': return_col
        } for expr_dict_original in expression_data  # 遍历每个原始表达式字典
        for scale_method in scale_method_sets  # 遍历每个放缩方法
        for roll_win in roll_win_sets  # 遍历每个滚动窗口
        for return_col in returns_columns
    ]

    process_list = split_k(k_split, res[0:10000])
    res1 = create_parellel(process_list=process_list,
                           callback=callback_fitness,
                           method=method,
                           fee=0.000005,
                           instruments=instruments,
                           total_data=total_data,
                           returns_data=returns_data)
    
    res1 = list(itertools.chain.from_iterable(res1))
    results = pd.DataFrame(res1)
    update_results(data=results,
                   keys=['name', 'scale_method', 'roll_win', 'ret_name'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--method',
                        type=str,
                        default='aicso0',
                        help='data method')
    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instruments')

    parser.add_argument('--task_id',
                        type=str,
                        default='200037',
                        help='code or instruments')

    args = parser.parse_args()
    test1(method=args.method,
          instruments=args.instruments,
          task_id=args.task_id)
