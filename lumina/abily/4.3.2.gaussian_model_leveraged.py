import os, pdb
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret


class GMMDiscretizer:
    """
    一个使用高斯混合模型(GMM)，将连续信号离散化或转换为概率加权仓位的转换器。
    它比K-Means更灵活，能适应非球形的簇分布。
    """

    def __init__(self, n_components: int = 3, random_state: int = 42):
        """
        初始化GMM离散化器。

        :param n_components: int, 混合模型中高斯分布的数量。对于多/空/平，通常为3。
        :param random_state: int, 随机种子以保证结果可复现。
        """
        if n_components != 3:
            raise ValueError("KMeansDiscretizer 目前只支持 n_clusters=3。")

        self.n_components = n_components
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            # 'full' 允许每个簇有自己的任意椭球协方差矩阵，最灵活
            covariance_type='full',
            # 增加迭代次数和初始化次数以获得更稳定的结果
            max_iter=200,
            n_init=10)

        # 存储学到的映射规则, e.g., {cluster_label_0: -1, ...}
        self.cluster_map_ = None
        # 存储排序后的簇中心均值，用于软分配
        self._sorted_means = None

    def fit(self, signal_series: pd.Series):
        """
        在给定的信号序列上学习标准化参数、GMM分布和映射规则。
        通常在 train_data + val_data 上调用。

        :param signal_series: pd.Series, 用于训练的连续元信号。
        """
        print("--- 正在使用GMM学习离散化规则 ---")
        if not isinstance(signal_series, pd.Series):
            raise TypeError("输入必须是一个Pandas Series。")

        # 1. 训练标准化器
        scaled_values = self.scaler.fit_transform(
            signal_series.values.reshape(-1, 1))

        # 2. 训练GMM模型
        self.gmm.fit(scaled_values)

        # 3. 确定从簇标签到{-1, 0, 1}的映射关系
        # 获取每个高斯分布的均值
        cluster_means = self.gmm.means_.flatten()
        self._sorted_means = np.sort(cluster_means)

        # 获取簇均值的排序索引
        sorted_center_indices = np.argsort(cluster_means)

        # 创建映射字典
        self.cluster_map_ = {
            sorted_center_indices[0]: -1,  # 均值最小的簇 -> -1 (做空)
            sorted_center_indices[1]: 0,  # 均值中间的簇 ->  0 (空仓)
            sorted_center_indices[2]: 1,  # 均值最大的簇 ->  1 (做多)
        }

        print("GMM离散化规则学习完成。")
        print(f"学习到的高斯分布均值 (标准化后): {self._sorted_means}")
        print(f"映射规则 (cluster_label -> position): {self.cluster_map_}")

        return self

    def transform(self,
                  signal_series: pd.Series,
                  soft_assignment: bool = False) -> pd.Series:
        """
        使用已经学习到的规则，对新的信号序列进行转换。

        :param signal_series: pd.Series, 待转换的连续元信号。
        :param soft_assignment: bool, 如果为True，则返回概率加权的连续仓位。
                                    如果为False，则返回硬分配的离散仓位{-1, 0, 1}。
        :return: pd.Series, 转换后的目标仓位。
        """
        if self.cluster_map_ is None:
            raise RuntimeError(
                "Discretizer has not been fitted yet. Call .fit() first.")

        # 1. 使用已经fit好的scaler进行标准化
        scaled_values = self.scaler.transform(
            signal_series.values.reshape(-1, 1))

        if soft_assignment:
            # --- 软分配：返回概率加权的连续仓位 ---
            print("执行软分配 (概率加权)...")
            # 获取每个点属于每个簇的概率
            probabilities = self.gmm.predict_proba(scaled_values)

            # 创建一个与概率矩阵列顺序一致的权重向量 {-1, 0, 1}
            # self.gmm.means_ 的顺序可能不是排序后的，所以要用cluster_map来正确赋权
            weights = np.array(
                [self.cluster_map_[i] for i in range(self.n_components)])

            # 矩阵乘法：(N, 3) * (3, 1) -> (N, 1)
            # 这等价于 P(空)*(-1) + P(平)*(0) + P(多)*(1)
            continuous_positions = np.dot(probabilities, weights)

            final_positions = pd.Series(continuous_positions,
                                        index=signal_series.index)
            final_positions.name = signal_series.name + '_gmm_soft'
        else:
            # --- 硬分配：返回离散的{-1, 0, 1}仓位 ---
            print("执行硬分配 (离散化)...")

            # 预测每个点属于哪个簇
            cluster_labels = self.gmm.predict(scaled_values)
            # 应用映射规则
            final_positions = pd.Series(cluster_labels,
                                        index=signal_series.index).map(
                                            self.cluster_map_)
            final_positions.name = signal_series.name + '_gmm_hard'

        return final_positions


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


def create_discretizer(positions_res, name):
    discretizer = GMMDiscretizer(n_components=3)
    train_positions = positions_res[name]['train']
    val_positions = positions_res[name]['val']
    test_positions = positions_res[name]['test']

    positions = pd.concat([train_positions, val_positions, test_positions],
                          axis=0)
    positions = positions.set_index('trade_time').sort_index()[name]

    discretizer.fit(train_positions.set_index('trade_time')[name])

    final_discrete_positions = discretizer.transform(positions)
    return final_discrete_positions


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