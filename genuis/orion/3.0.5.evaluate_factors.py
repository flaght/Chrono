import pdb, os, datetime, itertools, time, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from dotenv import load_dotenv

load_dotenv()

from lumina.genetic.util import create_id
from ultron.factor.genetic.geneticist.operators import calc_factor
from lib.ftd001 import fetch_temp_data1
from lib.cms003.metrics import Metrics
from kdutils.macro2 import *
from kdutils.tactix import Tactix
from kdutils.macro2 import base_path
from kdutils.process import split_k, run_process, create_parellel, add_process_env_sig


def merge_results(factor_names, results):
    flat_results = []
    for factor_name, df in zip(factor_names, results):
        # 将 DataFrame 拉平，列名会变成 (long, returns_mean), (topn, sharp) 这种多级列
        flat_series = df.unstack()

        # 把多级列名合并成单层列名，比如 'long_returns_mean', 'topn_sharp'
        flat_series.index = [f"{col[1]}_{col[0]}" for col in flat_series.index]

        # 加上因子名字
        flat_series.name = factor_name
        flat_results.append(flat_series)
    # 拼接成一个大表：每一行是一个因子，列包含了它所有长短多空指标
    final_wide_df = pd.DataFrame(flat_results)
    final_wide_df.index.name = "name"
    return final_wide_df


def create_params(params):
    m = hashlib.md5()
    # params可能是字典类型，需要转换为字符串
    if isinstance(params, dict):
        # 将字典按键排序后转换为字符串，确保相同参数组合产生相同hash
        params_str = str(sorted(params.items()))
    else:
        params_str = str(params)
    m.update(bytes(params_str, encoding='UTF-8'))
    return create_id(original=m.hexdigest(), digit=16)


def parallel_evaluate1(total_data, factor_name, ret_name, image_dirs):
    new_factor_name = "LAST('{}')".format(factor_name)
    result = Metrics.quick(returns=total_data[f'{ret_name}'],
                           factors=total_data[factor_name],
                           factor_name=new_factor_name,
                           hold=1,
                           skip=0,
                           category=1,
                           max_points=200,
                           save_file=os.path.join(
                               image_dirs,
                               f"{create_params(new_factor_name)}.png"),
                           quantiles=10,
                           is_series=True)

    result['name'] = new_factor_name
    result['id'] = create_params(new_factor_name)
    return result


def parallel_evaluate2(column, total_data1, returns_data, image_dirs):
    factor_name = column['forumla']
    factor_id = column['features']
    print(factor_name)
    dt = calc_factor(factor_name,
                     total_data=total_data1,
                     indexs=[],
                     key='code')
    dt = dt.set_index('code', append=True)['transformed'].unstack()
    result = Metrics.quick(returns=returns_data,
                           factors=dt,
                           factor_name=factor_name,
                           hold=1,
                           skip=0,
                           category=1,
                           max_points=200,
                           save_file=os.path.join(image_dirs,
                                                  f"{factor_id}.png"),
                           quantiles=10,
                           is_series=True)
    result['name'] = factor_name
    result['id'] = factor_id
    return result

def parallel_factors1(column, total_data1):
    factor_name = column['forumla']
    direction = column['direction']
    dt = calc_factor(factor_name,
                     total_data=total_data1,
                     indexs=[],
                     name=factor_name,
                     key='code')
    dt = dt.set_index('code', append=True)
    dt[factor_name] = dt[factor_name] * direction
    return dt
    

@add_process_env_sig
def run_evaluate2(target_column, total_data1, returns_data, image_dirs):
    results = run_process(target_column=target_column,
                          callback=parallel_evaluate2,
                          total_data1=total_data1,
                          returns_data=returns_data,
                          image_dirs=image_dirs)
    # result = parallel_evaluate2(total_data1=total_data1,
    #                             returns_data=returns_data,
    #                             factor_name=target_column['forumla'],
    #                             factor_id=target_column['features'],
    #                             image_dirs=image_dirs)
    return results

@add_process_env_sig
def run_factors1(target_column, total_data1):
    results = run_process(target_column=target_column,
                          callback=parallel_evaluate2,
                          total_data1=total_data1)
    return results


def load_derivate_factor(method, task_id, ret_name, sessions=[]):
    ## 加载挖掘表达式
    file_path = os.path.join(base_path, method, 'miner', task_id, ret_name)
    file_path = Path(file_path)
    res = []
    for feather_file in file_path.glob('*.feather'):
        data = pd.read_feather(feather_file)
        res.append(data)
    data1 = pd.concat(res, axis=0)
    return data1


## 衍生因评估
def derivate(method, task_id, ret_name):
    total_factors = fetch_temp_data1(method=method,
                                     task_id=task_id,
                                     datasets=['train', 'val'])
    total_returns = fetch_temp_data1(method=method,
                                     task_id=task_id,
                                     datasets=['train', 'val'],
                                     category='return')
    total_data = total_factors.merge(
        total_returns[['trade_time', 'code', f'{ret_name}']],
        on=['trade_time', 'code'])

    express_factors = load_derivate_factor(method=method,
                                           task_id=task_id,
                                           ret_name=ret_name)
    express_factors = express_factors.sort_values(by=['forumla'])
    factor_columns = express_factors[['forumla',
                                      'features']].to_dict(orient='records')
    total_data1 = total_data.sort_values(
        by=['trade_time', 'code']).set_index('trade_time')

    output_dirs = os.path.join(base_path, method, 'evaluate', str(task_id),
                               "derivative")
    image_dirs = os.path.join(output_dirs, "plot")
    os.makedirs(image_dirs, exist_ok=True)
    returns_data = total_data1.set_index('code',
                                         append=True)[ret_name].unstack()
    k_split = 2
    process_list = split_k(k_split, factor_columns)
    results = create_parellel(process_list=process_list,
                              callback=run_evaluate2,
                              total_data1=total_data1,
                              returns_data=returns_data,
                              image_dirs=image_dirs)
    results = list(itertools.chain.from_iterable(results))
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(output_dirs, "summary.csv"))


## 基础因子评估
def basic(method, task_id, ret_name):
    total_factors = fetch_temp_data1(method=method,
                                     task_id=task_id,
                                     datasets=['train', 'val'])
    total_returns = fetch_temp_data1(method=method,
                                     task_id=task_id,
                                     datasets=['train', 'val'],
                                     category='return')
    total_data = total_factors.merge(
        total_returns[['trade_time', 'code', f'{ret_name}']],
        on=['trade_time', 'code'])
    factor_columns = [
        f for f in total_data.columns
        if f not in ['trade_time', 'code', f'{ret_name}']
    ]
    factor_columns = factor_columns[:]
    total_data1 = total_data.set_index(['trade_time', 'code']).unstack()

    output_dirs = os.path.join(base_path, method, "evaluate",
                               TASK_MAPPING[task_id]['period'],
                               TASK_MAPPING[task_id]['source'], str(task_id))

    output_dirs = os.path.join(base_path, method, 'evaluate', str(task_id),
                               "basic")
    image_dirs = os.path.join(output_dirs, "plot")
    os.makedirs(image_dirs, exist_ok=True)

    results = Parallel(n_jobs=2, verbose=1)(
        delayed(parallel_evaluate1)(total_data=total_data1[[
            factor_columns[i],
            f'{ret_name}',
        ]],
                                    factor_name=factor_columns[i],
                                    image_dirs=image_dirs,
                                    ret_name=ret_name)
        for i in range(0, len(factor_columns)))
    #final_wide_pd = merge_results(factor_columns, results)
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(output_dirs, "summary.csv"))


def screening(method, task_id, ret_name):
    output_dirs = os.path.join(base_path, method, 'evaluate', str(task_id))
    basic_csv = pd.read_csv(os.path.join(output_dirs, "basic", "summary.csv"),
                            index_col=0)
    basic_csv['category'] = 'basic'

    derivative_csv = pd.read_csv(os.path.join(output_dirs, "derivative",
                                              "summary.csv"),
                                 index_col=0)
    derivative_csv['category'] = 'derivative'
    results = pd.concat([basic_csv, derivative_csv], axis=0)
    results['abs_ic'] = np.fabs(results['ic'])
    results = results.sort_values(by=['abs_ic'], ascending=False).dropna()
    results = results[(results['abs_ic'] > 0.1) & (results['abs_ic'] < 0.5) &
                      (results['turnover'] < 0.7)]
    results['factor_id'] = results['name'].apply(lambda x: create_params(x))
    return results


def create(method, task_id, session, ret_name):
    pdb.set_trace()
    dirs = os.path.join(base_path, method, 'evaluate', str(task_id), "results",
                        session)
    expression = pd.read_csv(os.path.join(dirs, "draft.csv"))
    ## 生成因子值 训练集 校验集 测试集
    factor_columns = expression.to_dict(orient='records')[0:3]
    pdb.set_trace()
    total_data = fetch_temp_data1(method=method,
                                     task_id=task_id,
                                     datasets=['val'])
    total_data1 = total_data.sort_values(
        by=['trade_time', 'code']).set_index('trade_time')
    
    k_split = 1
    process_list = split_k(k_split, factor_columns)
    
    results = create_parellel(process_list=process_list,
                              callback=run_evaluate2,
                              total_data1=total_data1)
    pdb.set_trace()
    print('-->')


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == 'basic':
        basic(method=variant.method,
              task_id=variant.task_id,
              ret_name=variant.ret_name)
    elif variant.form == 'derive':
        derivate(method=variant.method,
                 task_id=variant.task_id,
                 ret_name=variant.ret_name)
    elif variant.form == 'screen':
        screening(method=variant.method,
                  task_id=variant.task_id,
                  ret_name=variant.ret_name)
    elif variant.form == 'factor':
        create(method=variant.method,
               task_id=variant.task_id,
               session=variant.session,
               ret_name=variant.ret_name)
