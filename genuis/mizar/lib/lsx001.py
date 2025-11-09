import os, pdb, math, itertools
import pandas as pd
from ultron.factor.genetic.geneticist.operators import *
from kdutils.macro2 import *
from lib.iux001 import fetch_data, aggregation_data, fetch_times
from lib.aux001 import calc_expression


## 加载选中
def fetch_chosen_factors(method, instruments, task_id, period):
    filename = os.path.join(base_path, method, instruments, "rulex",
                            str(task_id), "nxt1_ret_{0}h".format(period),
                            "chosen.csv")
    expressions = pd.read_csv(filename).to_dict(orient='records')
    expressions = {item['formula']: item for item in expressions}
    expressions = list(expressions.values())
    return expressions


## 加载数据
def fetch_data1(method, task_id, instruments, datasets, period, expressions):
    total_data = fetch_data(method=method,
                            task_id=task_id,
                            instruments=instruments,
                            datasets=datasets)
    #program_list = list(expressions.keys())
    features = [
        eval(program['formula'])._dependency for program in expressions
    ]
    features = list(itertools.chain.from_iterable(features))
    features = list(set(features))
    total_data = total_data[['trade_time', 'code'] + features +
                            ['nxt1_ret_{}h'.format(period)]]
    return total_data


## 计算因子
def create_factors(total_data, expressions):
    res = []
    total_data1 = total_data.set_index('trade_time')
    #for program, direction in expressions.items():
    for expression in expressions:
        print(expression['formula'])
        factor_data = calc_expression(expression=expression['formula'],
                                      total_data=total_data1)
        factor_data['transformed'] = factor_data['transformed'] * expression[
            'direction']
        factor_data = factor_data.set_index(['trade_time', 'code'])

        factor_data.rename(columns={'transformed': expression['formula']},
                           inplace=True)
        res.append(factor_data)
    factors_data = pd.concat(res, axis=1)
    return factors_data


### 缺失数据前置填充
def build_factors(method,
                  instruments,
                  task_id,
                  period,
                  datasets=['train', 'val', 'test']):
    expressions = fetch_chosen_factors(method=method,
                                       instruments=instruments,
                                       task_id=task_id,
                                       period=period)
    total_data = fetch_data1(method=method,
                             task_id=task_id,
                             instruments=instruments,
                             datasets=datasets,
                             period=period,
                             expressions=expressions)
    factors_data = create_factors(total_data=total_data,
                                  expressions=expressions)
    
    factors_data = factors_data.unstack().fillna(method='ffill').stack()
    '''
    numeric_df = factors_data.select_dtypes(include=np.number)
    bad_values_mask = numeric_df.isnull() | np.isinf(numeric_df)
    bad_counts = bad_values_mask.sum()
    problematic_columns = bad_counts[bad_counts > 0]
    '''
    pdb.set_trace()
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs, "final_data.feather")
    final_data = factors_data.reset_index().merge(
        total_data[['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)]],
        on=['trade_time', 'code'])
    print(filename)
    final_data.to_feather(filename)


## 因子等权合成降频
def create_equal(factors_data, total_data, period):
    final_data = factors_data.mean(axis=1)
    final_data.name = 'transformed'
    final_data = aggregation_data(final_data.reset_index(), total_data, period)
    return final_data


####
def equal_weight(method,
                 instruments,
                 period,
                 expressions,
                 datasets=['train', 'val', 'test']):
    total_data = fetch_data1(method=method,
                             instruments=instruments,
                             datasets=datasets,
                             period=period,
                             expressions=expressions)
    factors_data = create_factors(total_data=total_data,
                                  expressions=expressions)
    final_data = create_equal(factors_data=factors_data,
                              total_data=total_data,
                              period=period)
    return final_data
