import os, itertools, joblib, pdb, json,hashlib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # 用于轮廓系数
from collections import namedtuple

from lumina.genetic.fusion.macro import EmpyricalTuple, AssignmentTuple, KMeansResultTuple
import ultron.factor.empyrical as empyrical
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_pnl
from lumina.genetic.process import *
from lumina.genetic.util import create_id

### 聚类合成信号


### 不同的信号映射组合 评估
def create_metrics(column, cluster_ids, positions_data, market_data,
                   strategy_setting):
    name = column.name
    params = column.params
    current_mapping = {
        cluster_id: signal_value
        for cluster_id, signal_value in zip(cluster_ids, params)
    }
    current_signals = positions_data['cluster'].map(current_mapping)
    current_signals = current_signals.to_frame()
    current_signals.columns = pd.MultiIndex.from_tuples([('pos', 'IM')])

    df = calculate_ful_ts_pnl(pos_data=current_signals,
                              total_data=market_data,
                              strategy_settings=strategy_setting)
    returns = df['ret']
    calmar_ratio = empyrical.calmar_ratio(returns=returns,
                                          period=empyrical.DAILY)
    sharpe_ratio = empyrical.sharpe_ratio(returns=returns,
                                          period=empyrical.DAILY)
    sortino_ratio = empyrical.sortino_ratio(returns=returns,
                                            period=empyrical.DAILY)
    max_drawdown = empyrical.max_drawdown(returns=returns)
    annual_return = empyrical.annual_return(returns=returns,
                                            period=empyrical.DAILY)
    annual_volatility = empyrical.annual_volatility(returns=returns,
                                                    period=empyrical.DAILY)

    metrics = EmpyricalTuple(name=name,
                             annual_return=annual_return,
                             annual_volatility=annual_volatility,
                             calmar=calmar_ratio,
                             sharpe=sharpe_ratio,
                             max_drawdown=max_drawdown,
                             sortino=sortino_ratio,
                             returns_series=returns)
    return KMeansResultTuple(name=name,
                             params=params,
                             mapping=current_mapping,
                             cluster=column.cluster,
                             empyrical=metrics)


@add_process_env_sig
def run_metrics(target_column, cluster_ids, positions_data, market_data,
                strategy_setting):
    metrics = run_process(target_column=target_column,
                          cluster_ids=cluster_ids,
                          callback=create_metrics,
                          positions_data=positions_data,
                          market_data=market_data,
                          strategy_setting=strategy_setting)
    return metrics


class Rotors(object):

    def __init__(self, signal_values, n_clusters, k_split=1):
        self.k_split = k_split
        self.stanard = len(signal_values)
        self.signal_values = signal_values
        self.n_clusters = n_clusters
        self.kmeans_optimal = KMeans(n_clusters=n_clusters,
                                     random_state=42,
                                     n_init=10)
        self.scaler = StandardScaler()

    def standard_data(self, positions_data):
        x_scaled = self.scaler.fit_transform(positions_data[['value']])
        positions_data['value_scaled'] = x_scaled

    def cluster_centers(self, positions_data, n_clusters):
        positions_data['cluster'] = self.kmeans_optimal.fit_predict(
            positions_data['value_scaled'].values.reshape(-1, 1))
        cluster_centers = self.kmeans_optimal.cluster_centers_.flatten()
        cluster_ids = range(n_clusters)
        return cluster_ids, cluster_centers

    def possible_signal(self, n_clusters, cluster_centers):
        if n_clusters == self.stanard:
            possible_signal_assignments = list(
                itertools.permutations(self.signal_values, n_clusters))
        elif n_clusters > self.stanard:
            # 更实际的方法：将聚类中心排序，然后将最小的映射为 -1，最大的映射为 1
            # 其他聚类根据其中心值的位置决定映射到 0 或其他值
            sorted_cluster_indices = np.argsort(cluster_centers)
            # 例如，将最小的映射为 -1，最大的映射为 1，中间的映射为 0
            base_mapping = {
                sorted_cluster_indices[0]: -1,
                sorted_cluster_indices[-1]: 1
            }
            # 其他中间的映射为 0
            for i in sorted_cluster_indices[1:-1]:
                base_mapping[i] = 0
            possible_signal_assignments = [tuple(base_mapping.values())
                                           ]  # 只考虑这一种基本映射

        return [
            AssignmentTuple(name="{0}_{1}_{2}_{3}".format(
                n_clusters, assignmet[0], assignmet[1], assignmet[2]),
                            params=assignmet,
                            cluster=n_clusters)
            for assignmet in possible_signal_assignments
        ]

    def evaluation(self, positions_data, market_data, strategy_setting):
        market_data['trade_vol', market_data['open'].columns[0]] = (
            strategy_setting['capital'] / market_data['open'] /
            strategy_setting['size'])

        self.standard_data(positions_data=positions_data)

        cluster_ids, cluster_centers = self.cluster_centers(
            positions_data=positions_data, n_clusters=self.n_clusters)

        possible_signal_assignments = self.possible_signal(
            n_clusters=self.n_clusters, cluster_centers=cluster_centers)

        process_list = split_k(self.k_split, possible_signal_assignments)

        res = create_parellel(process_list=process_list,
                              callback=run_metrics,
                              cluster_ids=cluster_ids,
                              positions_data=positions_data,
                              market_data=market_data,
                              strategy_setting=strategy_setting)
        res = list(itertools.chain.from_iterable(res))
        return res

    def save_model(self, path, best_mapping, strategies=None):
        if not os.path.exists(path):
            os.makedirs(path)

        ## 根据
        task_info = {
            'mapping':best_mapping
        }
        if strategies:
            task_info['strategies'] = [strategy._asdict() for strategy in strategies]
        s = hashlib.md5(
                json.dumps(task_info).encode(encoding="utf-8")).hexdigest()
        task_id = create_id(original=s, digit=10)
        model_path = os.path.join(path, 'strategy_{0}.pkl'.format(task_id))
        infos = {
            'optimal_model': self.kmeans_optimal,
            'scaler': self.scaler,
            'mapping': best_mapping
        }
        if strategies:
            infos['strategies'] = strategies
        joblib.dump(infos, model_path)


class Rotor(object):

    @classmethod
    def from_pickle(cls, path, name):
        model_path = os.path.join(path, 'strategy_{0}.pkl'.format(name))

        infos = joblib.load(model_path)
        strategies = infos.get('strategies', None)
        return cls(kmeans_optimal=infos['optimal_model'],
                   scaler=infos['scaler'],
                   best_mapping=infos['mapping'],
                   strategies=strategies)

    def __init__(self, kmeans_optimal, scaler, best_mapping, strategies=None):
        self._kmeans_optimal = kmeans_optimal
        self._scaler = scaler
        self._best_mapping = best_mapping
        self._strategies = strategies

    @property
    def strategies(self):
        return self._strategies
    
    def predict(self, positions_data):
        positions_data['value_scaled'] = self._scaler.transform(
            positions_data[['value']])
        positions_data['cluster'] = self._kmeans_optimal.predict(
            positions_data['value_scaled'].values.reshape(-1, 1))
        current_signals = positions_data['cluster'].map(self._best_mapping)
        return current_signals
