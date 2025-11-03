### 生成对比图
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
from lib.iux001 import fetch_data
from lib.iux002 import FactorComparator, calc_all

leg_mappping = {"rbb": ["hcb"], "ims": ["ics"]}


def create_evalute(column, period, factor_data, instruments, outputs):
    left_evaluate = calc_all(expression=column,
                             total_data1=factor_data,
                             period=period)
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


def fetch_chosen(method, instruments, task_id, period):
    
    filename = os.path.join(base_path, method, instruments, "rulex",
                            str(task_id), "nxt1_ret_{}h".format(str(period)),
                            "chosen.csv")
    chosen_data = pd.read_csv(filename)
    return chosen_data


def run2(method,
         instruments,
         period,
         task_id,
         session,
         datasets=['train', 'val']):
    left_symbol = instruments

    ## 优先创建目录，避免无判断没有跑过
    
    outputs = os.path.join("records", method, left_symbol, 'rulex',
                           str(task_id), "nxt1_ret_{}h".format(str(period)),
                           "d" + str(session))
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    
    ### 此目录为挖掘的原始目录
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

    formulas_in = chosen_data['formula']
    is_not_in_p2 = ~programs['formual'].isin(formulas_in)
    programs = programs[is_not_in_p2]

    programs['final_fitness'] = np.abs(programs['final_fitness'])
    
    programs = programs[programs['final_fitness'] > 0.03]

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
    process_list = split_k(k_split, expression_list)
    res = create_parellel(process_list=process_list,
                          callback=run_evalute,
                          period=period,
                          factor_data=factor_data,
                          instruments=instruments,
                          outputs=outputs)


if __name__ == '__main__':
    variant = Tactix().start()
    run2(method=variant.method,
         instruments=variant.instruments,
         period=variant.period,
         task_id=variant.task_id,
         session=variant.session)
