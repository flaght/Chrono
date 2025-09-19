### 通过对比校验的方式选择因子， 两个波动率相似的品种
import pdb, os, argparse, itertools
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

from ultron.factor.genetic.geneticist.operators import *
from lumina.genetic.process import *

from lib.aux001 import *
from lib.cux001 import *

from kdutils.common import *
from kdutils.macro2 import *
from kdutils.common import fetch_temp_data, fetch_temp_returns

##
leg_mappping = {"ims": ["ics"]}


def fetch_expression(dethod, method, instruments, period, task_id):
    #sessions = ['20250915', '20250916']
    sessions = ['20250917']
    res = []
    ## ./records/ic/bicso0/ims/evolution/nxt1_ret_15h/programs_200036_20250916.feather
    for session in sessions:
        filename = os.path.join(
            base_path, dethod, method, instruments, "evolution",
            "nxt1_ret_{0}h".format(period),
            "programs_{0}_{1}.feather".format(task_id, session))
        expression_data = pd.read_feather(filename)

        expression_data = expression_data.sort_values(by=['final_fitness'],
                                                      ascending=False)
        expression_data = expression_data[[
            'name', 'formual', 'final_fitness', 'raw_fitness'
        ]]
        res.append(expression_data)
    expression_data = pd.concat(res, axis=0).reset_index(drop=True)
    expression_data = expression_data[(expression_data['final_fitness'] > 0.03)
                                      & (expression_data['final_fitness'] < 1)]

    expression_data = expression_data.sort_values(by=['final_fitness'],
                                                  ascending=False)

    return expression_data


def fetch_data(method, instruments):
    total_factors = fetch_temp_data(method=method,
                                    instruments=instruments,
                                    datasets=['train', 'val', 'test'])

    total_returns = fetch_temp_returns(method=method,
                                       instruments=instruments,
                                       datasets=['train', 'val', 'test'],
                                       category='returns')
    total_data = total_factors.merge(total_returns, on=['trade_time', 'code'])
    return total_data


def create_perf(factor_data1, total_data1, period, expression):
    dt1 = factor_data1.reset_index().merge(total_data1.reset_index()[[
        'trade_time', 'code', 'nxt1_ret_{0}h'.format(period)
    ]],
                                           on=['trade_time', 'code'])
    is_on_mark = dt1['trade_time'].dt.minute % int(period) == 0
    dt1 = dt1[is_on_mark]
    evaluate1 = FactorEvaluate1(factor_data=dt1,
                                factor_name='transformed',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=240,
                                fee=0.000,
                                scale_method='roll_zscore',
                                expression=expression)
    status = evaluate1.run()
    status['expression'] = expression
    return status


def create_factors(expression, total_data1):
    factor_data1 = calc_factor(expression=expression,
                               total_data=total_data1,
                               indexs=[],
                               key='code')
    backup_cycle = 1
    factor_data1 = factor_data1.replace([np.inf, -np.inf], np.nan)
    factor_data1['transformed'] = np.where(
        np.abs(factor_data1.transformed.values) > 0.000001,
        factor_data1.transformed.values, np.nan)
    factor_data1 = factor_data1.loc[factor_data1.index.unique()[backup_cycle:]]
    return factor_data1


def calc_all(expression, total_data1, period):
    factor_data1 = create_factors(expression=expression,
                                  total_data1=total_data1)
    return create_perf(factor_data1=factor_data1,
                       total_data1=total_data1,
                       period=period,
                       expression=expression)


def run(dethod, method, instruments, period, task_id):
    expression_data = fetch_expression(dethod=dethod,
                                       method=method,
                                       instruments=instruments,
                                       period=period,
                                       task_id=task_id)
    left_data = fetch_data(method=method,
                           instruments=instruments).set_index('trade_time')
    right_data = fetch_data(
        method=method,
        instruments=leg_mappping[instruments][0]).set_index('trade_time')
    ##  遍历计算绩效
    res = []
    for expression in expression_data.itertuples():
        left_states = calc_all(expression=expression.formual,
                               total_data1=left_data,
                               period=period)
        right_states = calc_all(expression=expression.formual,
                                total_data1=right_data,
                                period=period)
        print("left: ic:{0} calmar:{1}".format(left_states['ic_mean'],
                                               left_states['calmar']))
        print("right: ic:{0} calmar:{1}".format(right_states['ic_mean'],
                                                right_states['calmar']))
        res.append(left_states)
        res.append(right_states)
    pdb.set_trace()
    print('-->')


def create_compare(column, period, left_data, right_data):
    left_states = calc_all(expression=column['formual'],
                           total_data1=left_data,
                           period=period)
    left_states['name'] = 'left'
    right_states = calc_all(expression=column['formual'],
                            total_data1=right_data,
                            period=period)
    right_states['name'] = 'right'
    return pd.DataFrame([left_states, right_states])


@add_process_env_sig
def run_compare(target_column, period, left_data, right_data):
    status_data = run_process(target_column=target_column,
                              callback=create_compare,
                              period=period,
                              left_data=left_data,
                              right_data=right_data)
    return status_data


def run1(dethod, method, instruments, period, task_id):
    expression_data = fetch_expression(dethod=dethod,
                                       method=method,
                                       instruments=instruments,
                                       period=period,
                                       task_id=task_id)
    left_data = fetch_data(method=method,
                           instruments=instruments).set_index('trade_time')
    right_data = fetch_data(
        method=method,
        instruments=leg_mappping[instruments][0]).set_index('trade_time')
    k_split = 4
    expression_dict = expression_data.to_dict(orient='records')[30:]
    process_list = split_k(k_split, expression_dict)
    res = create_parellel(process_list=process_list,
                          callback=run_compare,
                          period=period,
                          left_data=left_data,
                          right_data=right_data)
    pdb.set_trace()
    res1 = list(itertools.chain.from_iterable(res))
    dst1 = pd.concat(res1)
    dst1['abs_ic_mean'] = np.abs(dst1['ic_mean'])
    dst1 = dst1[(dst1['abs_ic_mean']>0.02) &(dst1['calmar']>3)]
    dst1 = dst1[['expression','name','calmar','ic_mean']].reset_index(drop=True)
    pdb.set_trace()
    print('-->')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--dethod', type=str, default='ic', help='data method')

    parser.add_argument('--method',
                        type=str,
                        default='bicso0',
                        help='data method')
    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instrument')

    parser.add_argument('--task_id',
                        type=str,
                        default='200036',
                        help='code or instruments')

    parser.add_argument('--period',
                        type=str,
                        default='15',
                        help='code or instruments')

    args = parser.parse_args()

    run1(dethod=args.dethod,
        method=args.method,
        instruments=args.instruments,
        period=args.period,
        task_id=args.task_id)
