import pdb, os
import pandas as pd
import numpy as np
from ultron.utilities.logger import kd_logger
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression  # 导入线性回归模型


def callback_relevance(best_programs, benchmark_warehouse, alpha):
    """
    向量化优化的相关性惩罚函数，用于处理大量策略。

    :param best_programs: list, 待惩罚的Program对象列表。
    :param benchmark_warehouse: pd.DataFrame, 基准策略的仓位DataFrame。
    :param alpha: float, 当前的相关性惩罚系数。
    :return: list, 更新了fitness后的Program对象列表。
    """
    if len(best_programs) <= 0:
        return []
    kd_logger.info(f"开始对 {len(best_programs)} 个因子进行向量化相关性惩罚...")
    # --- 步骤 1: 一次性准备所有待惩罚因子的数据 ---
    programs_to_penalize = []
    factors_to_penalize = []
    for prog in best_programs:
        if prog._factor_data is not None and not prog._factor_data.empty:
            programs_to_penalize.append(prog)

            factor_data = prog._factor_data.reset_index().set_index([
                'trade_time', 'code'
            ]).sort_index().rename(columns={'transformed': prog._name})
            factors_to_penalize.append(factor_data)
        else:
            # 对于无效的program，直接设置惩罚为0
            prog.penalty(penalty=0, max_corr=0, alpha=alpha)

    if len(factors_to_penalize) == 0:
        kd_logger.info("没有有效的因子数据需要进行惩罚。")
        return best_programs

    all_factors_df = pd.concat(factors_to_penalize, axis=1)

    # --- 步骤 2: 一次性对齐所有数据 ---
    # 合并所有待惩罚因子和基准因子
    total_data_for_corr = pd.concat([all_factors_df, benchmark_warehouse],
                                    axis=1)

    # 一次性处理缺失值
    aligned_data = total_data_for_corr.dropna()

    if aligned_data.empty or len(aligned_data) < 100:
        kd_logger.warning("对齐后数据不足100行，所有策略惩罚为0。")
        for prog in programs_to_penalize:
            prog.penalty(penalty=0, max_corr=0, alpha=alpha)
        return best_programs

    # --- 步骤 3: 一次性计算总相关性矩阵 ---
    # 这是最核心的性能提升点
    kd_logger.info("正在计算总相关性矩阵...")
    total_corr_matrix = aligned_data.corr(method='spearman')

    # --- 步骤 4: 矩阵化提取最大相关性 ---
    # 获取待惩罚策略的名称和基准策略的名称
    program_names = all_factors_df.columns
    benchmark_names = benchmark_warehouse.columns

    # 从总相关性矩阵中，切片出“待惩罚策略”与“基准策略”之间的相关性部分
    # .loc[index_labels, column_labels]
    cross_corr_matrix = total_corr_matrix.loc[program_names, benchmark_names]

    # 计算每个策略（每一行）与所有基准策略的最大绝对相关性
    max_correlations = cross_corr_matrix.abs().max(axis=1)  # axis=1 表示沿行操作

    # --- 步骤 5: 批量应用惩罚 ---
    new_programs = []
    kd_logger.info("正在应用惩罚...")
    for prog in programs_to_penalize:
        max_corr = max_correlations.get(prog._name, 0)  # 从Series中安全地获取值
        if np.isnan(max_corr):
            kd_logger.error(f"{prog._name} max_corr nan")
            continue
        penalty = alpha * max_corr
        prog.penalty(penalty=penalty, max_corr=max_corr, alpha=alpha)
        new_programs.append(prog)

    kd_logger.info("相关性惩罚完成。")
    return new_programs


'''
def callback_relevance(best_programs, benchmark_warehouse, alpha):
    new_best_programs = []
    for best_program in best_programs:
        # 应该使用当前 program 自己的 _factor_data
        if best_program._factor_data is None or best_program._factor_data.empty:
            # 如果没有因子值，跳过惩罚
            new_best_programs.append(best_program)
            continue

        factor_data = best_program._factor_data.copy()
        # 确保列名为'transformed'，以匹配后续的corr计算
        if 'transformed' not in factor_data.columns:
            # 假设只有一列，重命名它
            factor_data.rename(columns={factor_data.columns[0]: 'transformed'},
                               inplace=True)

        factor_data = factor_data.reset_index().set_index(
            ['trade_time', 'code']).sort_index()
        aligned_data = pd.concat(
            [factor_data['transformed'], benchmark_warehouse],
            axis=1).dropna()
        if aligned_data.empty or len(aligned_data) < 100:  # 增加鲁棒性检查
            best_program.penalty(penalty=0, max_corr=0, alpha=alpha)
            new_best_programs.append(best_program)
            continue

        correlations = aligned_data.corr(method='spearman')['transformed']
        max_corr = correlations.drop('transformed',
                                     errors='ignore').abs().max()
        max_corr = 0 if pd.isna(max_corr) else max_corr

        penalty = alpha * max_corr
        best_program.penalty(penalty=penalty, max_corr=max_corr, alpha=alpha)
        new_best_programs.append(best_program)

    return new_best_programs
'''
'''
def rolling_ic(factor_series, returns_series, rolling_window=60):
    ranked_features = factor_series.rank(method='first')
    ranked_return = returns_series.rank(method='first')
    rolling_ic = ranked_features.rolling(
        window=rolling_window,
        min_periods=int(rolling_window * 0.5)).corr(ranked_return)

    return rolling_ic.mean()
'''


def rolling_ic(factor_series, returns_series, rolling_window=60):
    """
    一个健壮的滚动IC计算函数，能正确处理NaN值。

    :param factor_series: pd.Series, 因子值，索引为MultiIndex(time, asset)。
    :param returns_series: pd.Series, 收益率，索引与因子对齐。
    :param rolling_window: int, 滚动窗口大小。
    :return: float, 滚动IC的均值。
    """
    # 步骤1: 将因子和收益率合并到一个DataFrame中，并丢弃任何一个为NaN的行
    # 这是关键！我们只在因子和收益率都存在的"数据对"上进行后续计算。
    df = pd.concat(
        [factor_series.rename('factor'),
         returns_series.rename('return')],
        axis=1)
    df.dropna(inplace=True)
    if df.empty or len(df) / len(factor_series) < 0.7:
        return 0.0  # 如果没有任何有效数据对，IC为0

    ranked_features = df['factor'].rank(method='first')
    ranked_return = df['return'].rank(method='first')
    rolling_ic = ranked_features.rolling(
        window=rolling_window,
        min_periods=int(rolling_window * 0.5)).corr(ranked_return)

    return rolling_ic.mean()


## 计算核心因子库
def sequential_gain(basic_factors,
                    returns_series,
                    ic_threshold,
                    corr_threshold,
                    gain_threshold,
                    existing_factors=None):
    '''
    通过序贯信息增益筛选法，从候选因子中挑选出一个低相关、高增益的因子库。

    :param basic_factors: DataFrame, 待筛选的候选因子。
    :param returns_series: Series, 对应的未来收益率。
    :param existing_factors_df: DataFrame, 可选，已有的因子库作为起点。
    :param ic_threshold: float, 因子独立表现的最低IC要求。
    :param corr_threshold: float, 新因子与已选因子库的最大相关性容忍度。
    :param info_gain_threshold: float, 正交化后信息保留率的最低要求。
    :return: DataFrame, 最终筛选出的因子库。
    '''
    valid_count = 0
    independent_ics = {}
    for factor_name in basic_factors.columns:
        ic = rolling_ic(basic_factors[factor_name], returns_series)
        if abs(ic) >= ic_threshold:
            kd_logger.debug("factor {0} 符合独立IC要求 ic:{1}".format(
                factor_name, ic))
            independent_ics[factor_name] = ic
        else:
            valid_count += 1
            kd_logger.debug("factor {0} 不符合独立IC要求 ic:{1}".format(
                factor_name, ic))

    kd_logger.info("fitness 共:{0} {1} 未达标 比例:{2} 分数:{3}".format(
        len(basic_factors.columns), valid_count,
        float(valid_count / len(basic_factors.columns)), ic_threshold))

    if not independent_ics:
        kd_logger.info("没有因子的独立IC超过阈值，筛选结束。")
        return pd.DataFrame()

    # 按IC绝对值从高到低排序
    sorted_factors = sorted(independent_ics.keys(),
                            key=lambda name: abs(independent_ics[name]),
                            reverse=True)

    # 2. 序贯筛选循环
    kd_logger.info("开始序贯筛选循环...")

    if existing_factors is not None and not existing_factors.empty:
        selected_factors = existing_factors.copy()
        kd_logger.info(f"从 {len(selected_factors.columns)} 个已存在因子开始。")
    else:
        # 如果没有已存在因子库，用表现最好的那个因子作为起点
        best_initial_factor = sorted_factors.pop(0)
        selected_factors = basic_factors[[best_initial_factor]].copy()
        kd_logger.debug(
            f"选择 '{best_initial_factor}' (IC={independent_ics[best_initial_factor]:.4f}) 作为初始因子。"
        )

    # 逐个考察剩余的候选因子
    corr_count = 0
    orth_count = 0
    total_count = 0
    for i, factor_name in enumerate(sorted_factors):
        kd_logger.debug(
            f"考察候选因子 {i+1}/{len(sorted_factors)}: '{factor_name}' (独立IC={independent_ics[factor_name]:.4f})"
        )

        total_count += 1
        f_candidate = basic_factors[factor_name]

        # a. 计算与现有精选库的最大相关性
        corrs = selected_factors.corrwith(f_candidate, method='spearman')
        max_abs_corr = corrs.abs().max()
        kd_logger.debug(f" {factor_name} - 与已选库的最大相关性: {max_abs_corr:.4f}")

        if max_abs_corr > corr_threshold:
            kd_logger.debug(
                f"{factor_name}  - 结果: 剔除 (相关性 > {corr_threshold})")
            corr_count += 1
            continue

        # b. 正交化，提取增量信息
        # 准备回归数据，对齐索引并删除NaN
        data_for_regression = pd.concat(
            [f_candidate.rename('y'), selected_factors], axis=1).dropna()
        y = data_for_regression['y']
        X = data_for_regression.drop(columns='y')

        if X.empty or len(X) < 200:  # 保证回归有足够样本
            orth_count += 1
            kd_logger.debug("{factor_name}  - 结果: 剔除 (回归样本不足)")
            continue

        model = LinearRegression(fit_intercept=False)  # 通常因子数据已中心化，可不加截距
        model.fit(X, y)

        # 残差 epsilon 就是增量信息
        epsilon = pd.Series(y - model.predict(X),
                            index=y.index,
                            name='residual')

        # c. 信息增益检验
        perf_original = abs(independent_ics[factor_name])
        perf_residual = abs(rolling_ic(epsilon, returns_series))

        if perf_original < 1e-6:
            retention_ratio = 0
        else:
            retention_ratio = perf_residual / perf_original

        if retention_ratio < gain_threshold:
            orth_count += 1
            kd_logger.debug(
                f"{factor_name} - 结果: 剔除 (信息保留率 < {gain_threshold:.0%})")
            continue

        selected_factors[factor_name] = f_candidate
    kd_logger.info("总计:{0}, 相关性未达标:{1} 分数:{2}, 正交未达标:{3} 分数:{4}".format(
        total_count, corr_count, corr_threshold, orth_count, gain_threshold))
    return selected_factors


class WareHouse(object):
    """
    一个动态管理和蒸馏因子库的类，结合了方案一和方案二的思想。
    """

    def __init__(self,
                 rootid,
                 n_benchmark_clusters=30,
                 distill_trigger_size=2):
        """
        初始化动态因子库。

        :param core_factors: DataFrame，包含永不改变的核心风格因子。
        :param n_benchmark_clusters: int，蒸馏后benchmark库的目标大小（簇的数量）。
        :param distill_trigger_size: int，当live_warehouse新增了多少个因子后触发蒸馏。
        """
        self._rootid = rootid
        self.permanent_core = None
        self.n_benchmark_clusters = n_benchmark_clusters
        self.distill_trigger_size = distill_trigger_size
        self._new_factors_since_last_distill = 0

        self.dirs = os.path.join("temp", "lumina", "warehose",
                                 str(self._rootid))
        if not os.path.exists(self.dirs):
            os.makedirs(self.dirs)
        ## 加载基准库
        ## 加载核心库
        self.benchmark_filename = os.path.join(self.dirs,
                                               "benchmark_warehouse.feather")
        self.permanent_filename = os.path.join(self.dirs,
                                               "permanent_core.feather")
        self.permanent_core = self.load_data(filename=self.permanent_filename)
        self.benchmark_warehouse = self.load_data(
            filename=self.benchmark_filename)

        if self.permanent_core is not None:
            kd_logger.info("成功加载核心库")

        if self.benchmark_factors is not None:
            kd_logger.info("成功加载基准库")

        if self.benchmark_warehouse is None and self.permanent_core is not None:
            self.benchmark_warehouse = self.permanent_core.copy()
            kd_logger.info("未找到历史基准库，使用永久核心库进行初始化。")

        # --- 4. 初始化性能优化的 live_warehouse 缓存 ---
        self._live_factors_list = []
        self._live_factor_names = set()

    def load_data(self, filename):
        return pd.read_feather(filename).set_index([
            'trade_time', 'code'
        ]) if filename and os.path.exists(filename) else None

    def set_initial_benchmark(self, core_factors):
        """如果没有任何历史库和核心库，允许外部在第一代后设置初始基准"""
        self.permanent_core = core_factors.copy()
        kd_logger.info("设置核心库")
        ## 保存核心库
        self.permanent_core.reset_index().to_feather(self.permanent_filename)
        # GP适应度函数实际参考的基准库，初始时就是核心库
        if self.benchmark_warehouse is None:
            kd_logger.info("设置基础库")
            self.benchmark_warehouse = self.permanent_core.copy()
        ## 核心库一般都是指定的，不是通常加载的。这里设计为了在没有核心库时候，用第一代特征作为核心库。

    @property
    def permanent_factors(self):
        return self.permanent_core

    @property
    def benchmark_factors(self) -> pd.DataFrame:
        """
        获取当前用于相关性计算的基准因子库。
        GP的适应度函数应该调用此方法。
        """
        return self.benchmark_warehouse

    def add_new_factor(self, new_factor_series):
        min_variance_threshold = 1e-5  ## 暂时设置经验值，可使用动态平滑方法
        """高效地将新因子添加到 live 缓存中"""
        factor_name = new_factor_series.name
        if factor_name in self._live_factor_names or factor_name in self.permanent_core.columns:
            return
        ## 正交处理
        aligned_data = pd.concat(
            [new_factor_series.rename('candidate'), self.benchmark_warehouse],
            axis=1,
            join='outer').fillna(0)
        X = aligned_data[self.benchmark_warehouse.columns]
        y = aligned_data['candidate']

        if X.empty or len(X) < 100:
            kd_logger.info("警告: 用于正交化的数据不足，操作取消。")
            return

        try:
            model = LinearRegression()
            model.fit(X, y)
            # 3. 计算残差 (ε)
            residuals = y - model.predict(X)
            if residuals.var() < min_variance_threshold:
                kd_logger.info(
                    f"警告: 正交化后的因子 '{new_factor_series.name}' 方差过小 ({residuals.var():.2e})，信息量不足，不予入库。"
                )
                return
            residuals.name = new_factor_series.name
        except Exception as e:
            kd_logger.info(
                f"错误: 在对因子 '{new_factor_series.name}' 进行正交化时发生错误: {e}")
            return

        new_factor_series = residuals

        # 将新因子加入live_warehouse
        #self.live_warehouse[factor_name] = new_factor_series
        self._live_factors_list.append(new_factor_series)
        self._live_factor_names.add(factor_name)
        self._new_factors_since_last_distill += 1

        # 检查是否需要触发蒸馏
        kd_logger.debug(
            "new_factors_since_last_distill:{0}, distill_trigger_size:{1}".
            format(self._new_factors_since_last_distill,
                   self.distill_trigger_size))
        if self._new_factors_since_last_distill >= self.distill_trigger_size:
            kd_logger.info(
                f"触发蒸馏！新增因子达到 {self._new_factors_since_last_distill} 个。\n--- 开始蒸馏流程 ---"
            )
            self.distill()
            self._new_factors_since_last_distill = 0

    def distill(self):
        """
        执行蒸馏过程，更新benchmark_warehouse。
        """

        # 1. 构建 live_warehouse DataFrame
        if not self._live_factors_list:
            kd_logger.info("Live-warehouse 缓存为空，无需蒸馏。")
            return

        live_warehouse = pd.concat(self._live_factors_list, axis=1)

        # 2. 构建总因子池
        pool_list = [live_warehouse]
        if self.benchmark_warehouse is not None:
            pool_list.append(self.benchmark_warehouse)
        if self.permanent_core is not None:
            pool_list.append(self.permanent_core)

        # 使用concat一次性合并，并处理重复列
        total_pool = pd.concat(pool_list, axis=1)
        total_pool = total_pool.loc[:, ~total_pool.columns.duplicated()]

        #total_pool = total_pool.unstack().fillna(method='ffill').dropna().stack()
        # 如果总因子数小于目标簇数，无需蒸馏，直接将所有因子作为基准
        if len(total_pool.columns) <= self.n_benchmark_clusters:
            self.benchmark_warehouse = total_pool.copy()
            self.benchmark_warehouse.reset_index().to_feather(
                self.benchmark_filename)
            kd_logger.info(
                f"总因子数 ({len(total_pool.columns)}) 不足目标簇数 ({self.n_benchmark_clusters})，无需蒸馏。已将所有因子更新为基准库并保存。"
            )
            return

        # 3. 健壮性处理：移除零方差列
        variances = total_pool.std()
        constant_columns = variances[variances.abs() <
                                     1e-10].index  # 使用一个小的阈值以处理浮点数精度问题
        if not constant_columns.empty:
            total_pool = total_pool.drop(columns=constant_columns)

        # 4. 计算距离矩阵并处理残余NaN
        corr_matrix = total_pool.corr(method='spearman').abs()
        distance_matrix = 1 - corr_matrix
        distance_matrix.fillna(1, inplace=True)

        # 5. 聚类和提取代表元
        clustering = AgglomerativeClustering(
            n_clusters=min(self.n_benchmark_clusters,
                           len(total_pool.columns)),  # 确保簇数不大于样本数
            metric='precomputed',
            linkage='average')
        labels = clustering.fit_predict(distance_matrix.to_numpy())

        representatives = []
        for i in range(clustering.n_clusters_):
            cluster_member_indices = np.where(labels == i)[0]
            if len(cluster_member_indices) == 0: continue
            cluster_distances = distance_matrix.iloc[cluster_member_indices,
                                                     cluster_member_indices]
            avg_distances_to_others = cluster_distances.mean(axis=1)
            medoid_name = avg_distances_to_others.idxmin()
            representatives.append(medoid_name)

        # 6. 更新并持久化 benchmark_warehouse
        self.benchmark_warehouse = total_pool[representatives].copy()
        self.benchmark_warehouse.reset_index().to_feather(
            self.benchmark_filename)
        kd_logger.info(f"保存更新后的基准库到: {self.benchmark_filename}")
        kd_logger.info(
            f"蒸馏完成。新的基准库包含 {len(self.benchmark_warehouse.columns)} 个代表性因子。")

        # 7. 清空 live 缓存
        self._live_factors_list = []
        self._live_factor_names = set()
