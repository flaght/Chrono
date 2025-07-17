import os, pdb
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret


class KMeansDiscretizer:
    """
    一个使用K-Means聚类，将连续信号离散化为{-1, 0, 1}的转换器。
    它在训练数据上学习聚类中心和映射规则，然后可以应用到新数据上。
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        if n_clusters != 3:
            raise ValueError("KMeansDiscretizer 目前只支持 n_clusters=3。")

        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             random_state=self.random_state,
                             n_init=10)  # n_init='auto' in newer sklearn

        # 这个字典将存储学到的映射规则, e.g., {cluster_label_0: -1, cluster_label_1: 1, ...}
        self.cluster_map_ = None

    def fit(self, signal_series: pd.Series):
        """
        在给定的信号序列上学习标准化参数、聚类中心和映射规则。
        通常在 train_data + val_data 上调用。

        :param signal_series: pd.Series, 用于训练的连续元信号。
        """
        print("--- 正在使用K-Means学习离散化规则 ---")
        # 1. 训练标准化器
        # 需要将Series转为 (n_samples, 1) 的形状
        scaled_values = self.scaler.fit_transform(
            signal_series.values.reshape(-1, 1))
        # 2. 训练K-Means模型
        self.kmeans.fit(scaled_values)

        # 3. 确定从簇标签到{-1, 0, 1}的映射关系 (核心逻辑)
        # 获取聚类中心
        cluster_centers = self.kmeans.cluster_centers_.flatten()

        # 获取簇中心的排序索引
        # np.argsort() 返回的是从小到大的值的索引
        sorted_center_indices = np.argsort(cluster_centers)
        # 创建映射字典
        # 最小的中心点 -> -1 (做空)
        # 中间的中心点 ->  0 (空仓)
        # 最大的中心点 ->  1 (做多)
        self.cluster_map_ = {
            sorted_center_indices[0]: -1,
            sorted_center_indices[1]: 0,
            sorted_center_indices[2]: 1,
        }

        print("K-Means离散化规则学习完成。")
        print(f"聚类中心: {np.sort(cluster_centers)}")
        print(f"映射规则 (cluster_label -> position): {self.cluster_map_}")

        return self

    def transform(self, signal_series: pd.Series) -> pd.Series:
        """
        使用已经学习到的规则，对新的信号序列进行离散化转换。

        :param signal_series: pd.Series, 待转换的连续元信号。
        :return: pd.Series, 离散化后的目标仓位{-1, 0, 1}。
        """
        pdb.set_trace()
        # 1. 使用已经fit好的scaler进行标准化
        scaled_values = self.scaler.transform(
            signal_series.values.reshape(-1, 1))

        # 2. 使用已经fit好的kmeans模型进行预测，得到簇标签
        cluster_labels = self.kmeans.predict(scaled_values)

        # 3. 应用映射规则，得到最终的离散仓位
        discrete_positions = pd.Series(cluster_labels,
                                       index=signal_series.index).map(
                                           self.cluster_map_)

        discrete_positions.name = signal_series.name + '_kmeans_discrete'
        return discrete_positions


def load_positions(base_dirs, key, names=[]):
    dirs = os.path.join(os.path.join(
        base_dirs, 'positions', key)) if isinstance(
            key, str) else os.path.join(
                os.path.join(base_dirs, 'positions', key))
    pdb.set_trace()
    positions_res = {}
    for name in names:
        train_positions = pd.read_feather(
            os.path.join(dirs, "{0}_train.feather".format(name)))

        val_positions = pd.read_feather(
            os.path.join(dirs, "{0}_val.feather".format(name)))

        test_positions = pd.read_feather(
            os.path.join(dirs, "{0}_test.feather".format(name)))
        positions_res[name] = {
            'train': train_positions,
            'val': val_positions,
            'test': test_positions
        }
    return positions_res


def load_data(mode='train'):
    filename = os.path.join(base_path, method, g_instruments, 'level2',
                            '{0}_data.feather'.format(mode))
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    return factors_data.set_index('trade_time')


def create_discretizer(positions_res, name):
    discretizer = KMeansDiscretizer(n_clusters=3)
    train_positions = positions_res[name]['train']
    val_positions = positions_res[name]['val']
    test_positions = positions_res[name]['test']

    positions = pd.concat([train_positions, val_positions, test_positions],
                          axis=0)
    positions = positions.set_index('trade_time').sort_index()[name]
    discretizer.fit(train_positions.set_index('trade_time')[name])

    final_discrete_positions = discretizer.transform(positions)
    return final_discrete_positions


def calcute_fitness(positions,
                    total_data,
                    strategy_settings,
                    base_dirs,
                    key=None):
    pdb.set_trace()
    save_positions = positions.copy()
    name = positions.name
    positions.name = 'pos'
    positions = positions.reset_index()
    positions['code'] = INSTRUMENTS_CODES[g_instruments]
    positions = positions.set_index(['trade_time', 'code']).unstack()
    pnl_in_window = calculate_ful_ts_ret(
        pos_data=positions,
        total_data=total_data,
        strategy_settings=strategy_settings,
        agg=True  # 确保按天聚合
    )

    ### 存储绩效
    dirs = os.path.join(os.path.join(base_dirs, 'returns', key)) if isinstance(
        key, str) else os.path.join(os.path.join(base_dirs, 'returns', key))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    print(dirs)
    pnl_in_window.reset_index().to_feather(
        os.path.join(dirs, "{0}.feather".format(name)))

    ### 存储仓位
    dirs = os.path.join(os.path.join(
        base_dirs, 'positions', key)) if isinstance(
            key, str) else os.path.join(
                os.path.join(base_dirs, 'positions', key))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    print(dirs)
    save_positions.reset_index().to_feather(
        os.path.join(dirs, "{0}.feather".format(name)))


if __name__ == '__main__':
    method = 'aicso2'
    instruments = 'ims'
    g_instruments = 'ims'
    task_id = '200037'

    key = 'tst5'

    strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[g_instruments]] * 0.05,
        'slippage': 0,  #SLIPPAGE_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[g_instruments]]
    }

    base_dirs = os.path.join(os.path.join('temp', "{}".format(method),
                                          task_id))

    positions_res = load_positions(
        base_dirs, key,
        ['equal_weight', 'train_fitness_weight', 'vol_inv_weight'])

    val_data = load_data(mode='val')
    train_data = load_data(mode='train')
    test_data = load_data(mode='test')
    total_data = pd.concat([train_data, val_data, test_data],
                           axis=0).sort_values(by=['trade_time'])

    total_data = total_data.copy().reset_index().set_index(
        ['trade_time', 'code']).unstack()

    equal_weight_kmeans_positions = create_discretizer(
        positions_res=positions_res, name='equal_weight')

    calcute_fitness(positions=equal_weight_kmeans_positions,
                    total_data=total_data,
                    strategy_settings=strategy_settings,
                    base_dirs=base_dirs,
                    key=key)


    train_fitness_kmeans_positions = create_discretizer(
        positions_res=positions_res, name='train_fitness_weight')

    calcute_fitness(positions=train_fitness_kmeans_positions,
                    total_data=total_data,
                    strategy_settings=strategy_settings,
                    base_dirs=base_dirs,
                    key=key)

    vol_inv_weight_kmeans_positions = create_discretizer(
        positions_res=positions_res, name='vol_inv_weight')

    calcute_fitness(positions=vol_inv_weight_kmeans_positions,
                    total_data=total_data,
                    strategy_settings=strategy_settings,
                    base_dirs=base_dirs,
                    key=key)