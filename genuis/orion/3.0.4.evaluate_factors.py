import pdb, os, datetime, time
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from dotenv import load_dotenv

load_dotenv()

from ultron.factor.genetic.geneticist.operators import calc_factor
from lib.ftd001 import fetch_temp_data1
#from lib.cms002 import Metrics, DALIY_PER_YEAR
from lib.cms003.metrics import Metrics
from kdutils.macro2 import *
from kdutils.tactix import Tactix
from kdutils.macro2 import base_path


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


def parallel_evaluate1(total_data, factor_name, ret_name, output_dirs,
                       image_dirs):
    result = Metrics.general(returns=total_data[f'{ret_name}'],
                             factors=total_data[factor_name],
                             dummy=None,
                             top_n=20,
                             hold=1,
                             skip=1,
                             freq=DALIY_PER_YEAR,
                             fee=0.000,
                             quantiles=10,
                             show_log=False,
                             is_series=True)

    factors_dirs = os.path.join(output_dirs, factor_name)
    os.makedirs(factors_dirs, exist_ok=True)
    result.save_results(base_output_dir=factors_dirs,
                        title_prefix=f"Factor Evaluation:{factor_name}",
                        image_export_dir=image_dirs)
    return result.to_dataframe()


def parallel_evaluate2(total_data1, returns_data, factor_name, factor_id,
                       image_dirs):
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
    return result


def load_derivate_factor(method, task_id, ret_name, category, sessions=[]):
    ## 加载挖掘表达式
    file_path = os.path.join(base_path, method, category, task_id, ret_name)
    file_path = Path(file_path)
    res = []
    for feather_file in file_path.glob('*.feather'):
        data = pd.read_feather(feather_file)
        res.append(data)
    data1 = pd.concat(res, axis=0)
    return data1


## 衍生因评估
def derivate(method, task_id, ret_name, category):
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
                                           ret_name=ret_name,
                                           category=category)
    express_factors = express_factors.sort_values(by=['forumla'])
    factor_columns = express_factors[['forumla',
                                      'features']].to_dict(orient='records')
    total_data1 = total_data.sort_values(
        by=['trade_time', 'code']).set_index('trade_time')

    output_dirs = os.path.join(base_path, method, 'evaluate', str(task_id),
                               "derivative")
    image_dirs = os.path.join(output_dirs, "plot")
    os.makedirs(image_dirs, exist_ok=True)
    results = Parallel(n_jobs=2, verbose=1)(delayed(parallel_evaluate2)(
        total_data1=total_data1,
        returns_data=total_data1.set_index('code',
                                           append=True)[ret_name].unstack(),
        factor_name=factor_columns[i]['forumla'],
        factor_id=factor_columns[i]['features'],
        image_dirs=image_dirs) for i in range(0, len(factor_columns)))
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(output_dirs, "summary.csv"))


## 基础因子评估
def basic(method, task_id, ret_name, category):
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

    output_dirs = os.path.join(base_path, method,
                               TASK_MAPPING[task_id]['source'], 'evaluate',
                               str(task_id), "basic")
    image_dirs = os.path.join(output_dirs, "plot")
    os.makedirs(output_dirs, exist_ok=True)
    results = Parallel(n_jobs=2, verbose=1)(
        delayed(parallel_evaluate1)(total_data=total_data1[[
            factor_columns[i],
            f'{ret_name}',
        ]],
                                    factor_name=factor_columns[i],
                                    image_dirs=image_dirs,
                                    ret_name=ret_name)
        for i in range(0, len(factor_columns)))
    final_wide_pd = merge_results(factor_columns, results)
    final_wide_pd.reset_index().to_csv(os.path.join(output_dirs,
                                                    "summary.csv"))


if __name__ == '__main__':
    variant = Tactix().start()
    derivate(method=variant.method,
             task_id=variant.task_id,
             ret_name=variant.ret_name,
             category='miner')
