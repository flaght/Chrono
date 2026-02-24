import pdb, os, datetime
import pandas as pd
from joblib import Parallel, delayed
from dotenv import load_dotenv

load_dotenv()

from lib.ftd001 import fetch_temp_data1
from lib.cms002 import Metrics, DALIY_PER_YEAR
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

def parallel_evaluate(total_data, factor_name, ret_name, output_dirs, image_dirs):
    result = Metrics.general(returns=total_data[f'{ret_name}'],
                             factors=total_data[factor_name],
                             dummy=None,
                             top_n=20,
                             hold=1,
                             skip=1,
                             freq=DALIY_PER_YEAR,
                             fee=0.000,
                             show_log=False,
                             is_series=True)

    factors_dirs = os.path.join(output_dirs, factor_name)
    os.makedirs(factors_dirs, exist_ok=True)
    result.save_results(base_output_dir=factors_dirs,
                        title_prefix=f"Factor Evaluation:{factor_name}",
                        image_export_dir=image_dirs)
    return result.to_dataframe()


def run(method, task_id, ret_name):
    pdb.set_trace()
    total_factors = fetch_temp_data1(method=method,
                                    task_id=task_id,
                                    source=TASK_MAPPING[task_id]['source'],
                                    datasets=['train', 'val'])
    total_returns = fetch_temp_data1(method=method,
                                    task_id=task_id,
                                    source=TASK_MAPPING[task_id]['source'],
                                    datasets=['train', 'val'],
                                    category='return')
    total_data = total_factors.merge(
        total_returns[['trade_time', 'code', f'{ret_name}']],
        on=['trade_time', 'code'])
    factor_columns = [f for f in total_data.columns if f not in ['trade_time','code',f'{ret_name}']]
    factor_columns = factor_columns[0:8]
    total_data1 = total_data.set_index(['trade_time', 'code']).unstack()
    
    output_dirs = os.path.join(base_path, method, "evaluate", TASK_MAPPING[task_id]['period'], TASK_MAPPING[task_id]['source'],
                               str(task_id))
    
    output_dirs = os.path.join(base_path, method, TASK_MAPPING[task_id]['source'], 'evaluate', str(task_id))
    image_dirs = os.path.join(output_dirs, "plot")
    os.makedirs(output_dirs, exist_ok=True)
    results = Parallel(n_jobs=1, verbose=1)(
        delayed(parallel_evaluate)(total_data=total_data1[[
            factor_columns[i],
            f'{ret_name}',
        ]],
                                   factor_name=factor_columns[i],
                                   output_dirs=output_dirs,
                                   image_dirs=image_dirs,
                                   ret_name=ret_name)
        for i in range(0, len(factor_columns)))
    final_wide_pd = merge_results(factor_columns, results)
    final_wide_pd.reset_index().to_csv(os.path.join(output_dirs,
                                                    "summary.csv"))
    
    
if __name__ == '__main__':
    variant = Tactix().start()
    run(method=variant.method,
        task_id=variant.task_id,
        ret_name=variant.ret_name)