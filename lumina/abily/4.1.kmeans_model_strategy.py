## 聚类法 将多个策略信号合成值转化为信号值
import pandas as pd
import numpy as np
import sqlalchemy as sa
import os, pdb, sys, json, joblib

from dotenv import load_dotenv

load_dotenv()
os.environ['INSTRUMENTS'] = 'ims'
g_instruments = os.environ['INSTRUMENTS']

from lumina.genetic import Thruster
from lumina.genetic import StrategyTuple
from lumina.genetic import Actuator
from lumina.genetic import mean_variance_builder
from lumina.genetic import Rotors

from kdutils.macro import *
from kdutils.file import fetch_file_data


def fetch_strategy(task_id, threshold=1.0):
    sql = """
        select name, formual, signal_method, signal_params, strategy_method, fitness, strategy_params from genetic_strategy where task_id={0} order by fitness desc limit 80
    """.format(task_id)
    engine = sa.create_engine(os.environ['DB_URL'])
    dt = pd.read_sql(sql=sql, con=engine)
    dt = dt[(dt['fitness'] > threshold) & (dt['fitness'] < 6)]
    dt = [StrategyTuple(**d1) for d1 in dt.to_dict(orient='records')]
    dt = [d1 for d1 in dt if 'MPWMA' not in d1.formual]
    return dt


## 计算策略的收益率 用于筛选
def create_metrics(k_split, strategies_dt, strategy_settings, total_data):
    thruster = Thruster(k_split=k_split)

    results = thruster.calculate(strategies_infos=strategies_dt,
                                 strategy_setting=strategy_settings,
                                 total_data=total_data)

    ## 根据绩效筛选结果
    metrics_dt = [{
        'name': r1.name,
        'annual_return': r1.annual_return,
        'annual_volatility': r1.annual_volatility,
        'calmar': r1.calmar,
        'sharpe': r1.sharpe,
        'max_drawdown': r1.max_drawdown,
        'sortino': r1.sortino
    } for r1 in results
                  if r1.sharpe > 0.8 and r1.calmar > 0.8 and r1.calmar < 6]

    metrics_names = {item['name'] for item in metrics_dt}
    filter_strategies = [s for s in strategies_dt if s.name in metrics_names]

    ## 筛选出来的策略， 和策略对应的绩效
    filter_metrics = [s for s in results if s.name in metrics_names]
    return filter_strategies, filter_metrics


### 各个策略的信号
def create_postions(k_split, filter_strategies, total_data):
    actuator = Actuator(k_split=k_split)

    strategies_data = actuator.calculate(strategies_infos=filter_strategies,
                                         total_data=total_data)
    return strategies_data


### 合成信号
def merge_signals(strategies_data, filter_strategies):
    actuator = Actuator(k_split=k_split)
    weights_data = actuator.fitness_weight(strategies_infos=filter_strategies)
    positions_data = actuator.fitness_signal(
        strategies_infos=filter_strategies,
        strategies_data=strategies_data,
        weights_data=weights_data)
    return positions_data, weights_data


def create_series(filter_metrics):
    res = []
    for metrics in filter_metrics:
        returns_series = metrics.returns_series
        returns_series.name = metrics.name
        res.append(returns_series)
    return res


def rolling_optimizer(filter_strategies, filter_metrics):
    returns_series = create_series(filter_metrics=filter_metrics)
    returns_series = pd.concat(returns_series, axis=1)
    lower_bound = 0.0  # 权重下限为0
    upper_bound = 1.0
    ### 第一次使用等权
    current_positions = np.ones(
        len(filter_strategies)) / len(filter_strategies)

    grouped = returns_series.groupby(level=0)
    window = 5
    for i, (ref_date, this_data) in enumerate(grouped):
        if i > (window - 1):
            start_idx = i - (window - 1)
            end_idx = i + 1
            past_window_data = returns_series.iloc[start_idx:end_idx]
            np_returns = past_window_data.values.T
            er = np.mean(np_returns, axis=1)
            cov = np.cov(np_returns)
            status2, feval2, weights2 = mean_variance_builder(
                er=er,
                risk_model=cov,
                turnover=0.9,
                target_vol=0.2,
                current_pos=current_positions,
                lbound=lower_bound,
                ubound=upper_bound)
            current_positions = weights2


def save_metrics(path, metrics):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, '{0}.pkl'.format(metrics.name))
    joblib.dump(metrics, filename)


if __name__ == '__main__':
    method = 'aicso2'
    k_split = 4
    task_id = INDEX_MAPPING[instruments_codes[g_instruments][0]]
    strategies_dt = fetch_strategy(task_id)
    total_data = fetch_file_data(base_path=base_path,
                                 method=method,
                                 g_instruments=g_instruments,
                                 datasets=['train_data', 'val_data'])
    total_data['trade_time'] = pd.to_datetime(total_data['trade_time'])

    strategy_settings = {
        'capital': 10000000,
        'commission': COST_MAPPING[instruments_codes[g_instruments][0]],
        'slippage': SLIPPAGE_MAPPING[instruments_codes[g_instruments][0]],
        'size': CONT_MULTNUM_MAPPING[instruments_codes[g_instruments][0]]
    }

    ## 筛选策略 #夏普大于1  卡玛大于1
    filter_strategies, filter_metrics = create_metrics(
        k_split=k_split,
        strategies_dt=strategies_dt,
        strategy_settings=strategy_settings,
        total_data=total_data)

    ## 计算仓位
    strategies_data = create_postions(k_split=k_split,
                                      filter_strategies=filter_strategies,
                                      total_data=total_data)

    ## 信号合并
    pdb.set_trace()
    positions_data, weights_data = merge_signals(strategies_data,
                                                 filter_strategies)

    market_data = total_data.set_index(['trade_time', 'code'])[[
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint', 'vwap'
    ]]
    market_data = market_data.unstack()

    positions_data.name = 'value'
    positions_data = positions_data.reset_index().set_index('trade_time')
    ## 模型训练
    rotor = Rotors(signal_values=[-1, 0, 1], k_split=4, n_clusters=3)
    res = rotor.evaluation(positions_data=positions_data,
                           market_data=market_data,
                           strategy_setting=strategy_settings)
    path = os.path.join(base_path, method, g_instruments, 'kmeans')
    for r in res:
        rotor.save_model(path=path,
                         best_mapping=r.mapping,
                         strategies=filter_strategies)
        save_metrics(path=os.path.join(base_path, method, g_instruments,
                                       'metrics'),
                     metrics=r)
    ##
    '''
    short1 = positions_data.quantile(1 / 3)
    long1 = positions_data.quantile(2 / 3)
    signals_data = positions_data.where(positions_data > long1,
                                        1).where(positions_data < short1,
                                                 -1).fillna(0)
    '''
