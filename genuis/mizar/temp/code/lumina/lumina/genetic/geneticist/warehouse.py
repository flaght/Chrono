import pdb, os
import pandas as pd
import numpy as np
import ultron.factor.empyrical as empyrical
from ultron.utilities.logger import kd_logger
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret


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
    kd_logger.info(f"开始对 {len(best_programs)} 个策略进行向量化相关性惩罚...")

    # --- 步骤 1: 一次性准备所有待惩罚策略的仓位数据 ---
    programs_to_penalize = []
    positions_to_penalize = []
    for prog in best_programs:
        if prog._position_data is not None and not prog._position_data.empty:
            programs_to_penalize.append(prog)
            # 假设 _position_data
            #pos_series = prog._position_data.iloc[:, 0].rename(prog._name)
            position_data = prog._position_data.reset_index().set_index([
                'trade_time', 'code'
            ]).sort_index().rename(columns={'transformed': prog._name})
            positions_to_penalize.append(position_data)
        else:
            # 对于无效的program，直接设置惩罚为0
            prog.penalty(penalty=0, max_corr=0, alpha=alpha)

    if not positions_to_penalize:
        kd_logger.info("没有有效的仓位数据需要进行惩罚。")
        return best_programs

    all_positions_df = pd.concat(positions_to_penalize, axis=1)

    # --- 步骤 2: 一次性对齐所有数据 ---
    # 合并所有待惩罚仓位和基准仓位
    total_data_for_corr = pd.concat([all_positions_df, benchmark_warehouse],
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
    program_names = all_positions_df.columns
    benchmark_names = benchmark_warehouse.columns

    # 从总相关性矩阵中，切片出“待惩罚策略”与“基准策略”之间的相关性部分
    # .loc[index_labels, column_labels]
    cross_corr_matrix = total_corr_matrix.loc[program_names, benchmark_names]

    # 计算每个策略（每一行）与所有基准策略的最大绝对相关性
    max_correlations = cross_corr_matrix.abs().max(axis=1)  # axis=1 表示沿行操作

    # --- 步骤 5: 批量应用惩罚 ---
    kd_logger.info("正在应用惩罚...")
    for prog in programs_to_penalize:
        max_corr = max_correlations.get(prog._name, 0)  # 从Series中安全地获取值
        penalty = alpha * max_corr
        prog.penalty(penalty=penalty, max_corr=max_corr, alpha=alpha)

    kd_logger.info("相关性惩罚完成。")
    return best_programs


'''
def callback_relevance(best_programs, benchmark_warehouse, alpha):
    new_best_programs = []
    count = 0
    for best_program in best_programs:
        count += 1
        # 应该使用当前 program 自己的 _factor_data
        if best_program._position_data is None or best_program._position_data.empty:
            # 如果没有因子值，跳过惩罚
            new_best_programs.append(best_program)
            continue

        kd_logger.info(f"因子惩罚处理，{count} / {len(best_programs)} name:{best_program._name}")
        position_data = best_program._position_data.copy()
        # 确保列名为'transformed'，以匹配后续的corr计算
        if 'transformed' not in position_data.columns:
            # 假设只有一列，重命名它
            position_data.rename(
                columns={position_data.columns[0]: 'transformed'},
                inplace=True)

        position_data = position_data.reset_index().set_index(
            ['trade_time', 'code']).sort_index()
        aligned_data = pd.concat(
            [position_data['transformed'], benchmark_warehouse],
            axis=1).dropna()
        if aligned_data.empty or len(aligned_data) < 100:  # 增加鲁棒性检查
            best_program.penalty(penalty=0, max_corr=0)
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


def sequential_gaind(candidate_positions, programs_data, total_data,
                     custom_params, fitness_threshold, corr_threshold,
                     gain_threshold):
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
    strategy_settings = custom_params['strategy_settings']
    independent_fitness = {}
    #programs_data1 = programs_data.copy().set_index('name')
    ## 不用单独计算 使用已经算好惩罚的fitness
    valid_count = 0
    for program in programs_data.itertuples():
        valid_count += 1
        if program.raw_fitness > fitness_threshold:
            valid_count -= 1
            independent_fitness[program.name] = program.raw_fitness

    kd_logger.info("fitness 共:{0}, {1}个未达标 比例:{2} 分数:{3}".format(
        len(programs_data), valid_count,
        float(valid_count / len(programs_data)), fitness_threshold))
    # 按fitness从高到低排序
    sorted_programs = sorted(independent_fitness.keys(),
                             key=lambda name: abs(independent_fitness[name]),
                             reverse=True)
    if len(sorted_programs) <= 0:
        return None

    #  2. 序贯筛选循环
    kd_logger.info("开始序贯筛选循环...")

    # 选择表现最好的那个策略作为起点
    best_name = sorted_programs.pop(0)
    best_initial_program = programs_data[programs_data["name"] ==
                                         best_name].iloc[0]
    best_initial_positions = candidate_positions[best_name]

    selected_poistions = pd.concat([best_initial_positions], axis=1)

    kd_logger.info(
        f"选择 '{best_name}' (原始Perf={independent_fitness[best_name]:.4f}) 作为初始策略。"
    )

    total_data1 = total_data.reset_index().set_index(['trade_time',
                                                      'code']).unstack()

    corr_count = 0
    orth_count = 0
    total_count = 0

    ##  逐个考察剩余的候选策略
    for i, name in enumerate(sorted_programs):
        ## 阈值过滤
        program = next(programs_data[programs_data.name.isin([name])].itertuples())
        if program.name not in independent_fitness:
            kd_logger.debug(f"{program.name} fitness 不满足阈值")
            continue
        total_count += 1
        kd_logger.debug(
            f"考察候选因子 {i+1}/{len(programs_data)}: '{program.name}' (独立FinalFitness={independent_fitness[program.name]:.4f})"
        )
        if program.name == best_initial_program['name']:
            continue

        ## 计算与现有精选组合的最大相关性
        current_position = candidate_positions[program.name]
        corrs = selected_poistions.corrwith(current_position,
                                            method='spearman')
        max_abs_corr = corrs.abs().max()
        kd_logger.debug(f" {program.name} - 与已选库的最大相关性: {max_abs_corr:.4f}")
        if max_abs_corr > corr_threshold:
            corr_count += 1
            kd_logger.info(
                f"{program.name}  - 结果: 剔除 (相关性 > {corr_threshold})")
            continue
        # b. 对仓位序列进行正交化
        data_for_regression = pd.concat(
            [current_position.rename('y'), selected_poistions],
            axis=1).dropna()
        y = data_for_regression['y']
        X = data_for_regression.drop(columns='y')

        if X.empty or len(X) < 200:  # 保证回归有足够样本
            kd_logger.info("{factor_name}  - 结果: 剔除 (回归样本不足)")
            continue

        model = LinearRegression(fit_intercept=False)  # 通常因子数据已中心化，可不加截距
        model.fit(X, y)

        # 残差仓位 epsilon 就是“独立的交易行为”
        residual_position = pd.Series(y - model.predict(X), index=y.index)

        ## 归一化
        # 计算残差序列的绝对值的最大值
        max_abs_residual = residual_position.abs().max()
        # 为了避免除以零
        if max_abs_residual > 1e-8:
            # 将所有残差值除以这个最大绝对值，从而将它们的范围缩放到 [-1, 1]
            normalized_residual_position = residual_position / max_abs_residual
        else:
            # 如果所有残差都接近于0，那么归一化后也都是0
            normalized_residual_position = residual_position.copy()

        # 残差的大小 大于1（比如1.45）直接被解释为持仓的“强度”或“名义杠杆”。一个1.45的残差仓位，在计算收益时 (1.45 * return)，其效果相当于一个用了1.45倍杠杆的+1仓位。
        # 测试在 “单位杠杆“的策略 只关心残差所代表的“方向”和“相对强度”**，但将其总的风险暴露（杠杆）控制在与原始策略相同的水平（最大为1倍）
        kd_logger.debug(
            f"{program.name}  - 残差仓位已归一化，原最大绝对值: {max_abs_residual:.4f}, 现最大绝对值: {normalized_residual_position.abs().max():.4f}"
        )

        # 格式转化
        normalized_residual_position = normalized_residual_position.to_frame(
            name='pos').unstack(level='code')

        df = calculate_ful_ts_ret(pos_data=normalized_residual_position,
                                  total_data=total_data1,
                                  strategy_settings=strategy_settings)

        ## 策略评估
        residual_fitness = empyrical.sharpe_ratio(returns=df['a_ret'],
                                                  period=empyrical.DAILY)
        retention_ratio = (abs(residual_fitness) /
                           (abs(program.raw_fitness) + 1e-8))
        if retention_ratio < gain_threshold:  ## 挖掘内部被选中但相关性有被剔除
            orth_count += 1
            kd_logger.info(
                f"{program.name} - 结果: 剔除 (信息保留率 < {gain_threshold:.0%})")
            continue
        selected_poistions[program.name] = current_position
    kd_logger.info("总计:{0}, 相关性未达标:{1} 分数:{2}, 正交未达标:{3} 分数:{4}".format(
        total_count, corr_count, corr_threshold, orth_count, gain_threshold))
    return selected_poistions


## 计算核心因子库
def sequential_gain(candidate_programs, total_data, strategy_settings,
                    fitness_threshold, corr_threshold, gain_threshold):
    """
    通过序贯信息增益筛选法，从候选策略中挑选出一个行为低相关、绩效高增益的策略组合。

    :param candidate_programs: list, 待筛选的Program对象列表。每个对象应有_raw_fitness和_position_data属性。
    :param returns_series: DataFrame, 用于回测的市场数据。
    :param custom_params: dict, 回测时需要的自定义参数。
    :param fitness_threshold: float, 策略独立表现的最低要求 (如Sharpe > 0.2)。
    :param corr_threshold: float, 新策略仓位与已选策略仓位的最大相关性容忍度。
    :param gain_threshold: float, 正交化后绩效保留率的最低要求。
    :return: list, 最终筛选出的Program对象列表。
    """
    independent_fitness = {}
    ## 不用单独计算 使用已经算好惩罚的fitness
    for program in candidate_programs:
        if program._final_fitness >= fitness_threshold:
            independent_fitness[program._name] = program._final_fitness

    # 按Ifitness从高到低排序
    sorted_programs = sorted(independent_fitness.keys(),
                             key=lambda name: abs(independent_fitness[name]),
                             reverse=True)

    #  2. 序贯筛选循环
    kd_logger.info("开始序贯筛选循环...")
    best_name = sorted_programs.pop(0)
    best_initial_program = [
        program for program in candidate_programs if program._name == best_name
    ][0]
    selected_poistions = pd.concat([best_initial_program.position_data],
                                   axis=1)
    total_data1 = total_data.reset_index().set_index(['trade_time',
                                                      'code']).unstack()
    ##  逐个考察剩余的候选策略
    for i, program in enumerate(candidate_programs):
        kd_logger.debug(
            f"考察候选因子 {i+1}/{len(candidate_programs)}: '{program._name}' (独立FinalFitness={independent_fitness[program._name]:.4f})"
        )
        if program._name == best_initial_program._name:
            continue
        ## 计算与现有精选组合的最大相关性
        corrs = selected_poistions.corrwith(program.position_data,
                                            method='spearman')
        max_abs_corr = corrs.abs().max()
        kd_logger.debug(f" {program._name} - 与已选库的最大相关性: {max_abs_corr:.4f}")
        if max_abs_corr > corr_threshold:
            kd_logger.info(
                f"{program._name}  - 结果: 剔除 (相关性 > {corr_threshold})")
            continue

        # b. 对仓位序列进行正交化
        data_for_regression = pd.concat(
            [program.position_data.rename('y'), selected_poistions],
            axis=1).dropna()
        y = data_for_regression['y']
        X = data_for_regression.drop(columns='y')

        if X.empty or len(X) < 200:  # 保证回归有足够样本
            kd_logger.info("f{program.name}  - 结果: 剔除 (回归样本不足)")
            continue

        model = LinearRegression(fit_intercept=False)  # 通常因子数据已中心化，可不加截距
        model.fit(X, y)

        # 残差仓位 epsilon 就是“独立的交易行为”
        residual_position = pd.Series(y - model.predict(X), index=y.index)
        residual_position = residual_position.to_frame(name='pos').unstack(
            level='code')
        df = calculate_ful_ts_ret(pos_data=residual_position,
                                  total_data=total_data1,
                                  strategy_settings=strategy_settings)
        ## 策略评估
        residual_fitness = empyrical.sharpe_ratio(returns=df['a_ret'],
                                                  period=empyrical.DAILY)
        retention_ratio = (abs(residual_fitness) /
                           (abs(program._final_fitness) + 1e-8))

        if retention_ratio < gain_threshold:
            kd_logger.info(
                f"{program._name} - 结果: 剔除 (信息保留率 < {gain_threshold:.0%})")
            continue
        selected_poistions[program._name] = program.position_data
    return selected_poistions


class WareHouse(object):
    """
    一个为策略挖掘设计的、动态、持久化、高性能的策略库管理器。
    它存储和管理的是策略的最终仓位序列，而非因子值。
    """

    def __init__(self,
                 rootid,
                 n_benchmark_clusters=30,
                 distill_trigger_size=2):
        """
        初始化动态策略库。

        :param core_factors: DataFrame，包含永不改变的核心风格因子。
        :param n_benchmark_clusters: int，蒸馏后benchmark库的目标大小（簇的数量）。
        :param distill_trigger_size: int，当live_warehouse新增了多少个因子后触发蒸馏。
        """
        self._rootid = rootid
        self.permanent_core = None
        self.n_benchmark_clusters = n_benchmark_clusters
        self.distill_trigger_size = distill_trigger_size
        self._new_positions_since_last_distill = 0

        self.dirs = os.path.join("temp", "lumina", "warehose", "strategy",
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

        if self.benchmark_warehouse is not None:
            kd_logger.info("成功加载基准库")

        if self.benchmark_warehouse is None and self.permanent_core is not None:
            self.benchmark_warehouse = self.permanent_core.copy()
            kd_logger.info("未找到历史基准库，使用永久核心库进行初始化。")

        # --- 4. 初始化性能优化的 live_warehouse 缓存 ---
        self._live_positions_list = []
        self._live_positions_names = set()

    def load_data(self, filename):
        return pd.read_feather(filename).set_index([
            'trade_time', 'code'
        ]) if filename and os.path.exists(filename) else None

    def set_initial_benchmark(self, core_positions):
        """如果没有任何历史库和核心库，允许外部在第一代后设置初始基准"""
        self.permanent_core = core_positions.copy()
        kd_logger.info("设置核心库")
        ## 保存核心库
        self.permanent_core.reset_index().to_feather(self.permanent_filename)
        # GP适应度函数实际参考的基准库，初始时就是核心库
        if self.benchmark_warehouse is None:
            kd_logger.info("设置基础库")
            self.benchmark_warehouse = self.permanent_core.copy()
        ## 核心库一般都是指定的，不是通常加载的。这里设计为了在没有核心库时候，用第一代特征作为核心库。

    @property
    def permanent_postions(self):
        return self.permanent_core

    @property
    def benchmark_positions(self) -> pd.DataFrame:
        """
        获取当前用于相关性计算的基准策略仓位
        GP的适应度函数应该调用此方法。
        """
        return self.benchmark_warehouse

    def add_new_position(self, new_position_series):
        #min_variance_threshold = 1e-5  ## 暂时设置经验值，可使用动态平滑方法
        """高效地将一个新策略的仓位序列（通常是多资产的平均仓位）添加到缓存中"""
        position_name = new_position_series.name
        if position_name in self._live_positions_names or position_name in self.permanent_core.columns:
            return
        '''
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

        new_factor_series = residuals
        '''

        # 将新因子加入live_warehouse
        #self.live_warehouse[factor_name] = new_factor_series
        self._live_positions_list.append(new_position_series)
        self._live_positions_names.add(position_name)
        self._new_positions_since_last_distill += 1

        # 检查是否需要触发蒸馏
        kd_logger.debug(
            "new_positions_since_last_distill:{0}, distill_trigger_size:{1}".
            format(self._new_positions_since_last_distill,
                   self.distill_trigger_size))
        if self._new_positions_since_last_distill >= self.distill_trigger_size:
            kd_logger.info(
                f"触发蒸馏！新增因子达到 {self._new_positions_since_last_distill} 个。\n--- 开始蒸馏流程 ---"
            )
            self.distill()
            self._new_positions_since_last_distill = 0

    def distill(self):
        """
        执行蒸馏过程，更新benchmark_warehouse。
        """

        # 1. 构建 live_warehouse DataFrame
        if not self._live_positions_list:
            kd_logger.info("Live-warehouse 缓存为空，无需蒸馏。")
            return

        live_warehouse = pd.concat(self._live_positions_list, axis=1)
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
                f"总策略数 ({len(total_pool.columns)}) 不足目标簇数 ({self.n_benchmark_clusters})，无需蒸馏。已将所有因子更新为基准库并保存。"
            )
            return

        # 3. 健壮性处理：移除零方差列
        variances = total_pool.std()
        constant_columns = variances[variances.abs() <
                                     1e-10].index  # 使用一个小的阈值以处理浮点数精度问题
        if not constant_columns.empty:
            kd_logger.warning(f"发现并移除 {len(constant_columns)} 个恒定仓位策略。")
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
            f"蒸馏完成。新的基准库包含 {len(self.benchmark_warehouse.columns)} 个代表性策略。")

        # 7. 清空 live 缓存
        self._live_factors_list = []
        self._live_factor_names = set()
