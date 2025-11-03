### 筛选因子
import pandas as pd
import numpy as np
import pdb, argparse
import os, pdb, math, itertools
from dotenv import load_dotenv

load_dotenv()
from kdutils.tactix import Tactix
from ultron.factor.genetic.geneticist.operators import *

from kdutils.macro2 import *
from lib.iux001 import fetch_data, aggregation_data
from lib.aux001 import calc_expression
from lib.cux001 import FactorEvaluate1

leg_mappping = {"rbb": ["hcb"], "ims": ["ics"]}


def load_factors(method,
                 instruments,
                 period,
                 task_id,
                 session,
                 category='gentic'):
    dirs = os.path.join(base_path, method, instruments, category, 'ic',
                        str(task_id), "nxt1_ret_{}h".format(str(period)),
                        str(session))
    filename = os.path.join(
        dirs, "programs_{0}_{1}.feather".format(str(task_id), str(session)))

    programs = pd.read_feather(filename)
    pdb.set_trace()
    programs = programs[programs['final_fitness'] > 0.02][[
        'name', 'formual', 'final_fitness'
    ]]
    return programs


def valid_programs(method,
                   instruments,
                   period,
                   task_id,
                   datasets,
                   features,
                   programs,
                   calmar=5,
                   sharpe2=1.5,
                   abs_ic=0.02):
    total_data = fetch_data(method=method,
                            task_id=task_id,
                            instruments=instruments,
                            datasets=datasets)
    total_data = total_data[['trade_time', 'code'] + features +
                            ['nxt1_ret_{}h'.format(period)]]
    total_data1 = total_data.set_index(['trade_time'])
    res = []
    i = 0
    for program in programs.itertuples():
        #if i > 20:
        #    break
        #i += 1
        print(program)
        factor_data = calc_expression(expression=program.formual,
                                      total_data=total_data1)
        dt = aggregation_data(factor_data=factor_data,
                              returns_data=total_data,
                              period=period)
        evaluate1 = FactorEvaluate1(factor_data=dt,
                                    factor_name='transformed',
                                    ret_name='nxt1_ret_{0}h'.format(period),
                                    roll_win=15,
                                    fee=0.000,
                                    scale_method='roll_zscore',
                                    expression=program.formual)
        state_dt = evaluate1.run()
        state_dt['name'] = program.name
        state_dt['expression'] = program.formual
        res.append(state_dt)
    perf_data = pd.DataFrame(res)[[
        'name', 'expression', 'ic_mean', 'calmar', 'sharpe2'
    ]]
    pdb.set_trace()
    perf_data['abs_ic'] = np.abs(perf_data['ic_mean'])
    perf_data = perf_data[(perf_data['calmar'] > calmar)
                          & (perf_data['sharpe2'] > sharpe2) &
                          (perf_data['abs_ic'] > abs_ic)]
    validated_programs = programs[programs.name.isin(
        perf_data.name.unique().tolist())]
    return validated_programs


def run(method,
        instruments,
        period,
        session,
        task_id,
        sategory,
        dategory,
        calmar,
        sharpe2,
        abs_ic,
        is_compare,
        datasets=['train', 'val']):
    programs = load_factors(method=method,
                            instruments=instruments,
                            period=period,
                            task_id=task_id,
                            session=session,
                            category=sategory)
    pdb.set_trace()
    features = [
        eval(program.formual)._dependency for program in programs.itertuples()
    ]
    features = list(itertools.chain.from_iterable(features))
    features = list(set(features))

    ## 优先创建目录，避免无判断没有跑过
    dirs = os.path.join(base_path, method, instruments, dategory, 'ic',
                        str(task_id), "nxt1_ret_{}h".format(str(period)),
                        str(session))

    if not os.path.exists(dirs):
        os.makedirs(dirs)


    validated_programs = valid_programs(
        method=method,
        task_id=task_id,
        instruments=leg_mappping[instruments][0]
        if is_compare else instruments,
        period=period,
        datasets=datasets,
        features=features,
        programs=programs,
        calmar=calmar,
        sharpe2=sharpe2,
        abs_ic=abs_ic)
    dirs = os.path.join(base_path, method, instruments, dategory, 'ic',
                        str(task_id), "nxt1_ret_{}h".format(str(period)),
                        str(session))

    pdb.set_trace()
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(
        dirs, "programs_{0}_{1}.feather".format(task_id, str(session)))
    validated_programs.reset_index(drop=True).to_feather(filename)


### 过滤挖掘后的因子
def run1(method,
         instruments,
         period,
         task_id,
         session,
         datasets=['train', 'val']):
    run(method=method,
        instruments=instruments,
        period=period,
        task_id=task_id,
        session=session,
        sategory='gentic',
        dategory='eligible',
        datasets=datasets,
        calmar=5,
        sharpe2=1.5,
        abs_ic=0.02,
        is_compare=False)


### 筛选后的因子和对应品种品种匹配
def run2(method,
         instruments,
         period,
         task_id,
         session,
         datasets=['train', 'val']):
    run(method=method,
        instruments=instruments,
        period=period,
        task_id=task_id,
        session=session,
        sategory='eligible',
        dategory='valid',
        datasets=datasets,
        calmar=3,
        sharpe2=1.0,
        abs_ic=0.02,
        is_compare=True)


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--method',
                        type=str,
                        default='cicso0',
                        help='data method')

    parser.add_argument('--task_id',
                        type=str,
                        default='200037',
                        help='task id')

    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instruments')

    parser.add_argument('--period', type=int, default=5, help='period')

    parser.add_argument('--session',
                        type=str,
                        default=202509226,
                        help='session')
    args = parser.parse_args()

    run1(method=args.method,
         instruments=args.instruments,
         period=args.period,
         task_id=args.task_id,
         session=args.session)
    '''
    variant = Tactix().start()
    if variant.form == 'first':
        run1(method=variant.method,
         instruments=variant.instruments,
         period=variant.period,
         task_id=variant.task_id,
         session=variant.session)
    elif variant.form == 'second':
        run2(method=variant.method,
         instruments=variant.instruments,
         period=variant.period,
         task_id=variant.task_id,
         session=variant.session)
