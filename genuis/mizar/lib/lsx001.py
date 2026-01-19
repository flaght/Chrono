import os, pdb, math, itertools
import pandas as pd
from ultron.factor.genetic.geneticist.operators import *
from kdutils.macro2 import *
from lib.iux001 import fetch_data, aggregation_data, fetch_times, merging_data1
from lib.aux001 import calc_expression
from lib.svx001 import scale_factors
from lib.cux001 import FactorEvaluate1

def fetch_draft_factors(method, instruments, task_id, period, name):
    pdb.set_trace()
    filename = os.path.join(base_path, method, instruments, "rulex",
                            str(task_id), "nxt1_ret_{0}h".format(period),
                            "draft.csv")
    expressions = pd.read_csv(filename).to_dict(orient='records')
    expressions = {item['formula']: item for item in expressions}
    expressions = list(expressions.values())
    return expressions

## 加载选中
def fetch_chosen_factors(method, instruments, task_id, period, name):
    filename = os.path.join(base_path, method, instruments, "rulex",
                            str(task_id), "nxt1_ret_{0}h".format(period),
                            "chosen_{0}.csv".format(name))
    pdb.set_trace()
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
        print(expression['formula'], expression['direction'])
        factor_data = calc_expression(expression=expression['formula'],
                                      total_data=total_data1)
        #### 放在标准化中处理 调整方向
        #factor_data['transformed'] = factor_data['transformed'] * expression[
        #    'direction']
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
                  name,
                  datasets=['train', 'val', 'test']):
    if name in ['draft']:
        expressions = fetch_draft_factors(method=method,
                                          instruments=instruments,
                                          task_id=task_id,
                                          period=period,
                                          name=name)
    else:
        expressions = fetch_chosen_factors(method=method,
                                           instruments=instruments,
                                           task_id=task_id,
                                           period=period,
                                           name=name)
    total_data = fetch_data1(method=method,
                             task_id=task_id,
                             instruments=instruments,
                             datasets=datasets,
                             period=period,
                             expressions=expressions)
    factors_data = create_factors(total_data=total_data,
                                  expressions=expressions)




    factors_data = factors_data.unstack().fillna(method='ffill').stack()
    ## 标准化 保持和绩效验证一直
    old_data = factors_data.copy()
    columns = factors_data.columns
    pdb.set_trace()
    ## 与评估因子时候一致，评估因子是评估标准化后的因子，故IC方向也是标准化后的IC方向
    for expression in expressions:
        scale_factors(predict_data=factors_data,
                      method='roll_zscore',
                      win=15,
                      factor_name=expression['formula'])
        factors_data[expression['formula']] = factors_data['transformed'] * expression['direction']
        factors_data.drop(['transformed'],axis=1, inplace=True)

    '''
    for col in columns:
        scale_factors(predict_data=factors_data,
                      method='roll_zscore',
                      win=15,
                      factor_name=col)
        factors_data[col] = factors_data['transformed']
        factors_data.drop(['transformed'], axis=1, inplace=True)
    '''
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
    filename = os.path.join(dirs, "final_{0}_data.feather".format(name))
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
