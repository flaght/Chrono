### 生成双品种绩效对比图
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
from lib.iux002 import FactorComparator, calc_all, calc_all1

leg_mappping = {"rbb": ["hcb"], "hcb": ["rbb"], "ims": ["ics"]}


def create_evalute(column, period, left_data, right_data, left_symbol,
                   right_symbol, outputs):

    try:
        left_evaluate = calc_all1(expression=column,
                                  total_data1=left_data,
                                  period=period)
        right_evaluate = calc_all1(expression=column,
                                   total_data1=right_data,
                                   period=period)
        fc = FactorComparator(eval_left=left_evaluate,
                              eval_right=right_evaluate,
                              left_name=left_symbol,
                              right_name=right_symbol,
                              expression=column)
        fc.plot_comparison()
        fc.save_results(base_output_dir=outputs)
    except Exception as e:
        print("===>{0}".format(str(e)))
        return {}


@add_process_env_sig
def run_evalute(target_column, period, left_data, right_data, left_symbol,
                right_symbol, outputs):
    status_data = run_process(target_column=target_column,
                              callback=create_evalute,
                              period=period,
                              left_data=left_data,
                              right_data=right_data,
                              left_symbol=left_symbol,
                              right_symbol=right_symbol,
                              outputs=outputs)
    return status_data


def load_factors(method,
                 instruments,
                 period,
                 task_id,
                 session,
                 category='gentic'):
    pdb.set_trace()
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
    mix_columns = list(set(features) & set(total_data.columns))
    if len(mix_columns) <= 0:
        print("not any features!!!!")
        return pd.DataFrame()

    total_data = total_data[['trade_time', 'code'] + mix_columns +
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
    pdb.set_trace()
    left_symbol = instruments
    right_symbol = leg_mappping[instruments][0]

    ## 优先创建目录，避免无判断没有跑过
    outputs = os.path.join("records", method, left_symbol, 'rulex',
                           str(task_id), "nxt1_ret_{}h".format(str(period)),
                           str(session))
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    programs = load_factors(method=method,
                            instruments=instruments,
                            period=period,
                            task_id=task_id,
                            session=session,
                            category='eligible')  # 经过一轮筛选因子进行对比
    if programs.empty:
        print("No factors data the criteria")
        return

    features = [
        eval(program.formual)._dependency for program in programs.itertuples()
    ]
    features = list(itertools.chain.from_iterable(features))
    features = list(set(features))

    left_data = fetch_data1(method=method,
                            instruments=left_symbol,
                            datasets=datasets,
                            features=features,
                            task_id=task_id,
                            period=period)

    right_data = fetch_data1(method=method,
                             instruments=right_symbol,
                             datasets=datasets,
                             features=features,
                             task_id=task_id,
                             period=period)
    #task_id = INDEX_MAPPING[INSTRUMENTS_CODES[instruments]]
    k_split = 4
    expression_list = programs['formual'].tolist()
    process_list = split_k(k_split, expression_list)
    res = create_parellel(process_list=process_list,
                          callback=run_evalute,
                          period=period,
                          left_data=left_data,
                          right_data=right_data,
                          left_symbol=left_symbol,
                          right_symbol=right_symbol,
                          outputs=outputs)



def run3(method,
         instruments,
         period,
         task_id,
         filename='choose.csv',
         datasets=['recent']):
    pdb.set_trace()
    left_symbol = instruments
    right_symbol = leg_mappping[instruments][0]
    
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
        
    ## 筛选为P的
    chosen_data = chosen_data[chosen_data['category']=='p']
    features = [
        eval(program.formula)._dependency
        for program in chosen_data.itertuples()
    ]
    
    features = list(itertools.chain.from_iterable(features))
    features = list(set(features))
    
    left_data = fetch_data1(method=method,
                            instruments=left_symbol,
                            datasets=datasets,
                            features=features,
                            task_id=task_id,
                            period=period)

    right_data = fetch_data1(method=method,
                             instruments=right_symbol,
                             datasets=datasets,
                             features=features,
                             task_id=task_id,
                             period=period)
        
        
    k_split = 4
    expression_list = chosen_data['formula'].tolist()
    process_list = split_k(k_split, expression_list)
    res = create_parellel(process_list=process_list,
                          callback=run_evalute,
                          period=period,
                          left_data=left_data,
                          right_data=right_data,
                          left_symbol=left_symbol,
                          right_symbol=right_symbol,
                          outputs=outputs)

    
    
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

    run2(method=args.method,
         instruments=args.instruments,
         period=args.period,
         task_id=args.task_id,
         session=args.session)
    '''
    variant = Tactix().start()
    if variant.form == 'all':
        run2(method=variant.method,
             instruments=variant.instruments,
             period=variant.period,
             task_id=variant.task_id,
             session=variant.session)
    if variant.form == 'recent':
        run3(method=variant.method,
             instruments=variant.instruments,
             period=variant.period,
             task_id=variant.task_id,
             filename=variant.filename)
