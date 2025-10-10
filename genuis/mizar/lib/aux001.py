### 加载因子文件
import os, pdb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultron.factor.genetic.geneticist.operators import calc_factor
from lumina.genetic.metrics.evaluate import FactorEvaluate
from kdutils.common import fetch_temp_data, fetch_temp_returns
'''
def fetch_expression_file(method):
    temp1 = os.path.join('temp', method, '200036', 'evolution',
                         'programs_{0}.feather'.format('200036'))
    ms1 = pd.read_feather(temp1).drop(['update_time'],
                                      axis=1)[['name', 'formual']]
    return ms1.rename(columns={'formual': 'expression'})
'''


### 读取 训练集 校验集，测试集的时间范围
def fetch_times(method, task_id, instruments):
    train_data = fetch_temp_data(method=method,
                                 task_id=task_id,
                                 instruments=instruments,
                                 datasets=['train'])
    val_data = fetch_temp_data(method=method,
                               task_id=task_id,
                               instruments=instruments,
                               datasets=['val'])
    test_data = fetch_temp_data(method=method,
                                task_id=task_id,
                                instruments=instruments,
                                datasets=['test'])
    return {
        'train_time':
        (train_data['trade_time'].min(), train_data['trade_time'].max()),
        'val_time':
        (val_data['trade_time'].min(), val_data['trade_time'].max()),
        'test_time':
        (test_data['trade_time'].min(), test_data['trade_time'].max())
    }


### 读取 相关数据
def fetch_market(instruments, method, task_id, datasets):
    factors_data = fetch_temp_data(
        method=method,
        instruments=instruments,
        task_id=task_id,
        datasets=datasets if isinstance(datasets, list) else [datasets])

    returns_data = fetch_temp_returns(
        method=method,
        instruments=instruments,
        datasets=datasets if isinstance(datasets, list) else [datasets],
        category='returns')
    return factors_data.sort_values(
        by=['trade_time', 'code']), returns_data.sort_values(
            by=['trade_time', 'code'])


### 公式因子值计算
def calc_expression(expression, total_data):
    factor_data = calc_factor(expression=expression,
                              total_data=total_data,
                              key='code',
                              indexs=[])
    factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
    factor_data['transformed'] = np.where(
        np.abs(factor_data.transformed.values) > 0.000001,
        factor_data.transformed.values, np.nan)
    ##前置填充
    #factor_data = factor_data.assign(transformed=factor_data.groupby('code')
    #                                 ['transformed'].ffill()).dropna()
    factor_data = factor_data.loc[factor_data.index.unique()[1:]]
    factors_data1 = factor_data.reset_index()
    return factors_data1


### 绩效计算
def calc_evaluate(factor_data,
                  ret_name,
                  roll_win,
                  scale_method,
                  fee=0,
                  is_plot=False):
    evaluate = FactorEvaluate(
        factor_data=factor_data,
        factor_name='transformed',
        ret_name=ret_name,
        roll_win=roll_win,  # 因子放缩窗口，自定义
        fee=fee,
        scale_method=scale_method)  # 可换 'roll_zscore' 等
    result = evaluate.run()
    perf_data = evaluate.factor_data[[
        'ic', 'gross_ret', 'turnover', 'net_ret', 'nav'
    ]]
    if is_plot:
        plot_evaluate(perf_data, result, ret_name)
    return result, evaluate.factor_data[[
        'ic', 'gross_ret', 'turnover', 'net_ret', 'nav', 'f_scaled'
    ]]


### 绘图
def plot_evaluate(perf_data, result, ret_name):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    # 1. 净值 + 累计毛收益
    axes[0, 0].plot(perf_data['nav'], label='Net Asset Value')
    cum_gross = (1 + perf_data['gross_ret']).cumprod()
    axes[0, 0].plot(cum_gross, label='Cumulative Gross Return')
    axes[0, 0].set_title('Net Asset Value vs Cumulative Gross Return')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=30)

    # 2. 滚动 IC
    axes[0, 1].plot(perf_data['ic'])
    axes[0, 1].axhline(perf_data['ic'].mean(), color='red', ls='--')
    axes[0, 1].set_title(f'Rolling IC (mean={perf_data["ic"].mean():.3f})')
    axes[0, 1].tick_params(axis='x', rotation=30)

    # 3. 散点（无时间轴，无需旋转）
    sns.scatterplot(x='f_scaled',
                    y=ret_name,
                    data=ret_name,
                    ax=axes[1, 0],
                    s=10,
                    alpha=0.6)
    axes[1, 0].axhline(0, ls='--', c='grey')
    axes[1, 0].axvline(0, ls='--', c='grey')
    axes[1, 0].set_title('Factor vs Forward Return')

    # 4. 换手率
    axes[1, 1].plot(ret_name['turnover'])
    axes[1, 1].set_title(f'Turnover (mean={ret_name["turnover"].mean():.3f})')
    axes[1, 1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()


## 相似两个品种在同一个因子下对比
def both_compare(codes, expression, method, name=['train', 'val', 'test']):
    total_data0, total_returns0 = fetch_market(instruments=codes[0],
                                               method=method,
                                               task_id='200036',
                                               name=name)
    total_data1, total_returns1 = fetch_market(instruments=codes[1],
                                               method=method,
                                               task_id='140001',
                                               name=name)

    factors_data0 = calc_expression(expression,
                                    total_data0.set_index('trade_time'))
    factors_data1 = calc_expression(expression,
                                    total_data1.set_index('trade_time'))

    basic_data0, _ = calc_evaluate(factor_data=factors_data0.merge(
        total_returns0, on=['trade_time', 'code']),
                                   ret_name='time_weight',
                                   roll_win=60,
                                   scale_method='roll_min_max',
                                   fee=0,
                                   is_plot=False)
    basic_data1, _ = calc_evaluate(factor_data=factors_data1.merge(
        total_returns1, on=['trade_time', 'code']),
                                   ret_name='time_weight',
                                   roll_win=60,
                                   scale_method='roll_min_max',
                                   fee=0,
                                   is_plot=False)
    basic_data0['code'] = codes[0]
    basic_data1['code'] = codes[1]
    return pd.DataFrame([basic_data0, basic_data1])
