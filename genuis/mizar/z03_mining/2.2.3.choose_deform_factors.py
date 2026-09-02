### 过滤掉已经选中的因子生成绩效图
import pandas as pd
import numpy as np
import pdb, argparse
import os, pdb, math, itertools
from dotenv import load_dotenv

load_dotenv()
from kdutils.tactix import Tactix
from ultron.factor.genetic.geneticist.operators import *
from lumina.genetic.process import *
from kdutils.macro2 import *
from lib.aux001 import extract_operators
from lib.iux001 import fetch_data
from lib.iux002 import FactorComparator, calc_all1

leg_mappping = {"rbb": ["hcb"], "ims": ["ics"]}


def create_evalute(column, period, factor_data, instruments, outputs):
    left_evaluate = calc_all1(expression=column,
                              total_data1=factor_data,
                              period=period)
    filename = os.path.join(outputs, left_evaluate.name, "evaluation_plot.png")
    if os.path.exists(filename):
        print("{0} exists".format(filename))
    left_evaluate.run()
    left_evaluate.plot_results()
    left_evaluate.save_results(base_output_dir=outputs)


@add_process_env_sig
def run_evalute(target_column, period, factor_data, instruments, outputs):
    status_data = run_process(target_column=target_column,
                              callback=create_evalute,
                              period=period,
                              factor_data=factor_data,
                              instruments=instruments,
                              outputs=outputs)
    return status_data


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

    if not os.path.exists(filename):
        print("No factors file the criteria")
        return pd.DataFrame()
    programs = pd.read_feather(filename)

    programs = programs[programs['final_fitness'] > 0.02][[
        'name', 'formual', 'final_fitness'
    ]]
    return programs


def fetch_data1(method, instruments, datasets, features, task_id, period):
    total_data = fetch_data(method=method,
                            instruments=instruments,
                            task_id=task_id,
                            datasets=datasets)
    total_data = total_data[['trade_time', 'code'] + features +
                            ['nxt1_ret_{}h'.format(period)]]
    return total_data


def fetch_chosen(method, instruments, task_id, period, filename="choose.csv"):

    filename = os.path.join(base_path, method, instruments, "rulex",
                            str(task_id), "nxt1_ret_{}h".format(str(period)),
                            filename)
    print(filename)
    return pd.read_csv(filename) if os.path.exists(
        filename) else pd.DataFrame()


def run2(method,
         instruments,
         period,
         task_id,
         session,
         datasets=['train', 'val']):
    left_symbol = instruments
    pdb.set_trace()
    ## 优先创建目录，避免无判断没有跑过
    outputs = os.path.join("records", method, left_symbol, 'rulex',
                           str(task_id), "nxt1_ret_{}h".format(str(period)),
                           "d" + str(session))
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    ### 此目录为挖掘的原始目录 session 对应的目录
    programs = load_factors(method=method,
                            instruments=instruments,
                            period=period,
                            task_id=task_id,
                            session=session,
                            category='eligible')
    if programs.empty:
        print("No factors data the criteria")
        return

    ## 加载已经选中的因子
    chosen_data = fetch_chosen(method=method,
                               instruments=instruments,
                               task_id=task_id,
                               period=period)
    pdb.set_trace()
    ### 过滤掉在此session 已经被选中的因子
    if not chosen_data.empty:
        formulas_in = chosen_data['formula']
        is_not_in_p2 = ~programs['formual'].isin(formulas_in)
        programs = programs[is_not_in_p2]

    programs['final_fitness'] = np.abs(programs['final_fitness'])

    programs = programs[programs['final_fitness'] > 0.02]

    features = [
        eval(program.formual)._dependency for program in programs.itertuples()
    ]
    features = list(itertools.chain.from_iterable(features))
    features = list(set(features))

    factor_data = fetch_data1(method=method,
                              instruments=instruments,
                              datasets=datasets,
                              features=features,
                              task_id=task_id,
                              period=period)
    ### 过滤 不符合标准因子
    #task_id = INDEX_MAPPING[INSTRUMENTS_CODES[instruments]]
    k_split = 2
    expression_list = programs['formual'].tolist()
    expression_list = [
        expression for expression in expression_list
        if len(extract_operators(expression)) < 5
    ]
    pdb.set_trace()
    process_list = split_k(k_split, expression_list)
    res = create_parellel(process_list=process_list,
                          callback=run_evalute,
                          period=period,
                          factor_data=factor_data,
                          instruments=instruments,
                          outputs=outputs)


def run3(method,
         instruments,
         period,
         task_id,
         filename='choose.csv',
         datasets=['recent']):
    ## 加载初选目录
    outputs = os.path.join("records", method, instruments, 'rulex',
                           str(task_id), "nxt1_ret_{}h".format(str(period)),
                           "recent")
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    ## 会把选中的特征全部读取处理进行绘图
    chosen_data = fetch_chosen(method=method,
                               instruments=instruments,
                               task_id=task_id,
                               period=period,
                               filename=filename)
    features = [
        eval(program.formula)._dependency
        for program in chosen_data.itertuples()
    ]
    features = list(itertools.chain.from_iterable(features))
    features = list(set(features))
    factor_data = fetch_data1(method=method,
                              instruments=instruments,
                              datasets=datasets,
                              features=features,
                              task_id=task_id,
                              period=period)
    k_split = 4
    expression_list = chosen_data['formula'].tolist()
    expression_list = [
        expression for expression in expression_list
        if len(extract_operators(expression)) < 5
    ]
    process_list = split_k(k_split, expression_list)
    res = create_parellel(process_list=process_list,
                          callback=run_evalute,
                          period=period,
                          factor_data=factor_data,
                          instruments=instruments,
                          outputs=outputs)


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == 'all':
        run2(method=variant.method,
             instruments=variant.instruments,
             period=variant.period,
             task_id=variant.task_id,
             session=variant.session)
    elif variant.form == 'recent':
        run3(method=variant.method,
             instruments=variant.instruments,
             period=variant.period,
             task_id=variant.task_id,
             filename=variant.filename)
