import pdb, empyrical
import numpy as np
import pandas as pd
import os, pdb, sys, json, math
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression  # 导入线性回归模型

from dotenv import load_dotenv

load_dotenv()

from ultron.utilities.logger import kd_logger
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret
from kdutils.macro2 import *


## 加载总览fitness
def load_fitness(base_dirs):
    fitness_file = os.path.join(base_dirs, "fitness.feather")
    fitness_pd = pd.read_feather(fitness_file)

    return fitness_pd


## 加载不同时段的仓位数据
def load_positions(base_dirs, names):
    dirs = os.path.join(os.path.join(base_dirs, 'positions'))
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


def load_data(instruments, mode='train'):
    filename = os.path.join(base_path, method, instruments, 'level2',
                            '{0}_data.feather'.format(mode))
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    return factors_data.set_index('trade_time')


def merge_positions(positions_res, mode):
    res = []
    for name in positions_res:
        print(name)
        positions = positions_res[name][mode]
        positions = positions.rename(columns={'pos': name})
        res.append(positions.set_index('trade_time'))
    positions = pd.concat(res, axis=1).reset_index()
    return positions


class HelperStrategy(object):

    def __init__(self, instruments, task_id, config):
        self.instruments = instruments
        self.task_id = task_id
        self.config = config

    def stage_rolling_stability_check(self, programs: pd.DataFrame,
                                      positions: pd.DataFrame,
                                      total_data: pd.DataFrame,
                                      custom_params: dict,
                                      config: dict) -> pd.DataFrame:

        positions['code'] = INSTRUMENTS_CODES[self.instruments]
        positions = positions.set_index(['trade_time', 'code'])

        kd_logger.info("STAGE 3: Rolling Stability Check")
        initial_count = len(positions)
        # 准备 unstacked 的市场数据，以备循环内使用
        total_data = total_data.copy().reset_index().set_index(
            ['trade_time', 'code']).unstack()

        strategy_settings = custom_params.get('strategy_settings', {})
        res1 = []
        for name in programs['name']:
            kd_logger.info(f"  - Analyzing rolling stability for '{name}'...")
            # 获取单个策略的仓位，并 unstack
            position = positions[[name]].rename(columns={
                name: 'pos'
            }).unstack()
            daily_returns = calculate_ful_ts_ret(
                pos_data=position,
                total_data=total_data,
                strategy_settings=strategy_settings,
                agg=True  # 确保按天聚合
            )

            # 步骤B: 在日度收益率序列上进行滚动夏普计算
            # empyrical.roll_sharpe_ratio 是一个现成的、高效的函数
            rolling_sharpes = empyrical.roll_sharpe_ratio(
                daily_returns['a_ret'], window=config.get('rolling_window', 5))

            # empyrical的滚动函数可能不会使用step，如果需要带步长的滚动，
            # 我们可以手动实现或对结果进行抽样
            step = config.get('rolling_step', 20)
            if step > 1:
                rolling_sharpes = rolling_sharpes.iloc[::step]

            if rolling_sharpes.empty or rolling_sharpes.isnull().all():
                kd_logger.warning(f"  - Strategy '{name}': 滚动夏普计算失败。剔除。")
                continue

            rolling_mean = rolling_sharpes.mean()
            rolling_std = rolling_sharpes.std()
            win_rate = (rolling_sharpes > 0).mean()
            min_perf = rolling_sharpes.min()
            res1.append({
                'name': name,
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std,
                'win_rate': win_rate,
                'min_perf': min_perf,
            })

        return pd.DataFrame(res1)


class FilterStrategy(object):

    def __init__(self, instruments, task_id, config):
        self.instruments = instruments
        self.task_id = task_id
        self.config = config

    ## 快速清理掉大量不合格的策略 既要保证基础能力，也要看初步泛化
    def stage_sanity_check(self, programs, config):
        kd_logger.info("STAGE 1: Sanity Check")
        programs_pd = programs.copy()

        initial_count = len(programs_pd)
        programs_pd.replace([np.inf, -np.inf], 0, inplace=True)
        programs_pd.dropna(
            subset=['train_fitness', 'val_fitness', 'test_fitness'],
            inplace=True)

        programs_pd = programs_pd[programs_pd['train_fitness'] >
                                  config['min_train_fitness']]
        programs_pd = programs_pd[programs_pd['val_fitness'] >
                                  config['min_val_fitness']]

        kd_logger.info(f"Stage 1后剩余: {len(programs_pd)} / {initial_count}")
        return programs_pd

    # 筛选过拟合 考察样本外稳定性，对抗过拟合的核心。检查训练集到验证集的绩效衰减率。
    def stage_overfitting_check(self, programs, config):
        kd_logger.info("STAGE 2: Overfitting Check")

        programs_pd = programs.copy()

        initial_count = len(programs_pd)

        programs_pd = programs_pd[programs_pd['val_retention'] >
                                  config['min_retention_rate']]

        kd_logger.info(f"Stage 2后剩余: {len(programs_pd)} / {initial_count}")
        return programs_pd

    ## 训练集 + 验证集  滚动计算
    ## 在日度数据上进行滚动
    ## rolling_window和step的单位是交易日
    ## 考察**中长期（月/季度级别）**的稳定性
    ## 高，通过“先聚合再滚动”的策略，避免了冗余计算
    def stage_rolling_stability_check(self, programs: pd.DataFrame,
                                      positions: pd.DataFrame,
                                      total_data: pd.DataFrame,
                                      custom_params: dict,
                                      config: dict) -> pd.DataFrame:

        positions['code'] = INSTRUMENTS_CODES[self.instruments]
        positions = positions.set_index(['trade_time', 'code'])

        kd_logger.info("STAGE 3: Rolling Stability Check")
        initial_count = len(positions)
        # 准备 unstacked 的市场数据，以备循环内使用
        total_data = total_data.copy().reset_index().set_index(
            ['trade_time', 'code']).unstack()

        stable_survivors_names = []
        strategy_settings = custom_params.get('strategy_settings', {})
        res1 = []
        for name in programs['name']:
            kd_logger.info(f"  - Analyzing rolling stability for '{name}'...")
            # 获取单个策略的仓位，并 unstack
            position = positions[[name]].rename(columns={
                name: 'pos'
            }).unstack()
            daily_returns = calculate_ful_ts_ret(
                pos_data=position,
                total_data=total_data,
                strategy_settings=strategy_settings,
                agg=True  # 确保按天聚合
            )

            # 步骤B: 在日度收益率序列上进行滚动夏普计算
            # empyrical.roll_sharpe_ratio 是一个现成的、高效的函数
            rolling_sharpes = empyrical.roll_sharpe_ratio(
                daily_returns['a_ret'], window=config.get('rolling_window', 5))

            # empyrical的滚动函数可能不会使用step，如果需要带步长的滚动，
            # 我们可以手动实现或对结果进行抽样
            step = config.get('rolling_step', 20)
            if step > 1:
                rolling_sharpes = rolling_sharpes.iloc[::step]

            if rolling_sharpes.empty or rolling_sharpes.isnull().all():
                kd_logger.warning(f"  - Strategy '{name}': 滚动夏普计算失败。剔除。")
                continue

            rolling_std = rolling_sharpes.std()
            win_rate = (rolling_sharpes > 0).mean()
            min_perf = rolling_sharpes.min()
            # 步骤C: 应用筛选标准
            if (rolling_std < config['max_rolling_std']
                    and win_rate > config['min_rolling_win_rate']
                    and min_perf > config['min_worst_window_perf']):
                stable_survivors_names.append(name)
                kd_logger.info(f"  - Strategy '{name}': 通过滚动稳定性检验。")
            else:
                kd_logger.info(
                    f"  - Strategy '{name}': 滚动稳定性不达标。Std={rolling_std:.2f}, WinRate={win_rate:.2%}, MinPerf={min_perf:.2f}。剔除。"
                )

        survivors_df = programs[programs['name'].isin(stable_survivors_names)]
        kd_logger.info(
            f"Stage 3 (Rolling Stability) 后剩余: {len(survivors_df)} / {initial_count}"
        )
        return survivors_df

    # 基于行为聚类的邻里审查 (最核心的鲁棒性检验)。
    # 通过对策略在验证集上的仓位序列进行DBSCAN聚类，识别出稳定、高质量的“行为簇”，
    #  并剔除行为孤立的“噪声策略”和位于劣质簇中的策略。 如果全部孤立，说明所有策略之间没有相关性

    def stage_neighborhood_analysis(self, programs, positions, config):
        kd_logger.info("STAGE 3: Neighborhood Analysis")
        positions_pd = positions.copy()
        programs_pd = programs.copy()
        initial_count = len(programs_pd)
        if initial_count < config.get('min_cluster_input_size', 10):
            kd_logger.warning(f"进入第3关的策略数 ({initial_count})过少，跳过聚类分析。")
            return programs_pd

        # --- 1. 准备用于聚类的数据 ---
        # 确保仓位数据的列顺序与programs_df中的'name'一致
        positions_to_cluster = positions_pd[programs_pd['name'].tolist()]

        # 健壮性处理：移除恒定仓位策略（零方差），它们无法计算相关性
        variances = positions_to_cluster.std()
        constant_strategies = variances[variances.abs() < 1e-10].index

        if not constant_strategies.empty:
            kd_logger.warning(
                f"发现并移除 {len(constant_strategies)} 个恒定仓位策略: {constant_strategies.tolist()}"
            )
            positions_to_cluster.drop(columns=constant_strategies,
                                      inplace=True)
            # 同样从主df中移除
            programs_pd = programs_pd[~programs_pd['name'].
                                      isin(constant_strategies)]

        if len(programs_pd) < config.get('min_cluster_input_size', 10):
            kd_logger.warning("移除恒定仓位策略后，数量过少，跳过聚类分析。")
            return programs_pd

        # --- 2. 计算距离矩阵 ---
        kd_logger.info("正在计算策略间的行为距离矩阵...")
        # 使用 abs(corr) 是因为强正相关和强负相关都意味着行为上的强关联性
        corr_matrix = positions_to_cluster.corr(method='spearman').abs()
        distance_matrix = 1 - corr_matrix
        # 填充可能因计算问题产生的NaN
        distance_matrix.fillna(1.0, inplace=True)

        # --- 3. 执行DBSCAN聚类 ---
        kd_logger.info(
            f"执行DBSCAN聚类，eps={config['dbscan_eps']}, min_samples={config['dbscan_min_samples']}..."
        )

        clustering = DBSCAN(
            eps=config['dbscan_eps'],
            min_samples=config['dbscan_min_samples'],
            metric='precomputed',
            n_jobs=-1  # 使用所有可用CPU核心
        )

        labels = clustering.fit_predict(
            distance_matrix.loc[positions_to_cluster.columns,
                                positions_to_cluster.columns])
        # 将簇标签（家庭ID）添加回DataFrame
        # 创建一个Series来映射 name -> cluster_id
        cluster_map = pd.Series(labels,
                                index=positions_to_cluster.columns,
                                name='behavior_cluster_id')
        programs_pd = programs_pd.merge(cluster_map,
                                        left_on='name',
                                        right_index=True,
                                        how='left')

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        kd_logger.info(f"聚类完成: 发现 {n_clusters} 个行为簇，以及 {n_noise} 个噪声策略。")

        if n_clusters == 0 and n_noise == len(programs_pd):
            kd_logger.warning("所有策略都被识别为行为独立的噪声点。这可能表明运行时去相关非常成功。")
            kd_logger.warning("将所有策略视为通过本关，直接进入下一阶段筛选。")
            # 直接返回所有通过前两关的策略，不进行任何淘汰
            return programs_pd

        # --- 4. 筛选第一步：直接剔除噪声策略 ---
        survivors = programs_pd[programs_pd['behavior_cluster_id'] != -1]
        kd_logger.info(f"剔除 {n_noise} 个行为孤立的噪声策略后，剩余: {len(survivors)}")
        if survivors.empty:
            return pd.DataFrame()

        # --- 5. 筛选第二步：审查每个“行为簇”的质量 ---
        # 计算每个簇的统计特性
        cluster_stats = survivors.groupby(
            'behavior_cluster_id')['val_fitness'].agg(['mean', 'std', 'count'])
        cluster_stats.rename(columns={
            'mean': 'cluster_mean_val_fit',
            'std': 'cluster_std_val_fit',
            'count': 'cluster_size'
        },
                             inplace=True)

        # 计算变异系数 (CV) 来衡量簇的内部稳定性
        cluster_stats['cluster_cv'] = cluster_stats['cluster_std_val_fit'] / (
            cluster_stats['cluster_mean_val_fit'].abs() + 1e-8)

        # 找出“高质量”的簇
        good_clusters = cluster_stats[
            (cluster_stats['cluster_size'] >= config['min_cluster_size'])
            & (cluster_stats['cluster_mean_val_fit'] >
               config['min_cluster_fitness'])
            & (cluster_stats['cluster_cv'] < config['max_cluster_cv'])]

        kd_logger.info(
            f"共有 {len(cluster_stats)} 个簇，其中 {len(good_clusters)} 个被认定为高质量稳定簇。")
        if good_clusters.empty:
            return pd.DataFrame()

        # 只保留那些位于“高质量”簇中的策略
        final_survivors = survivors[survivors['behavior_cluster_id'].isin(
            good_clusters.index)]

        # 将簇的统计信息合并回来，供后续分析
        final_survivors = final_survivors.merge(good_clusters,
                                                left_on='behavior_cluster_id',
                                                right_index=True,
                                                how='left')

        kd_logger.info(
            f"Stage 3后最终剩余: {len(final_survivors)} / {initial_count}")
        return final_survivors

    # 排序是为了选择，选择过程不能看最终答案。 验证集相关指标，对策略进行综合评分和排序
    def stage_quality_scoring(self, programs, config):
        kd_logger.info("STAGE 4: Quality Scoring")

        programs_pd = programs.copy()

        # 使用排名法，并赋予样本外和稳定性更高的权重
        # 1. 计算基于验证集指标的排名分数
        # rank(pct=True) -> 值越大，排名越高 (0到1)
        programs_pd['rank_val_fit'] = programs_pd['val_fitness'].rank(pct=True)

        # val_retention 是保留率，越大越好。所以直接用rank即可。
        # rank(pct=True) -> val_retention越大，排名越高，分数也越高
        programs_pd['score_val_stable'] = programs_pd['val_retention'].rank(
            pct=True)

        programs_pd['quality_score'] = (
            config['quality_score_weights']['val_fit'] *
            programs_pd['rank_val_fit'] +
            config['quality_score_weights']['val_stable'] *
            programs_pd['score_val_stable'])

        kd_logger.info("已根据验证集表现和保留率计算出 quality_score。")

        programs_pd = programs_pd[programs_pd['quality_score'] >=
                                  config['quality_score_weights']
                                  ['min_quality_score']]
        programs_pd = programs_pd.sort_values('quality_score', ascending=False)
        return programs_pd

    def stage_sequential_selection(self, programs: pd.DataFrame,
                                   positions: pd.DataFrame,
                                   total_data: pd.DataFrame,
                                   custom_params: dict,
                                   config: dict) -> pd.DataFrame:
        """
        关卡五：团队选拔 - 序贯绩效增益筛选。
        严格只在验证集上进行，以挑选出行为多样化、信息互补的最终策略组合。
        """
        kd_logger.info("STAGE 5: Sequential Selection on Validation Set")

        total_data1 = total_data.reset_index().set_index(
            ['trade_time', 'code']).unstack()
        positions['code'] = INSTRUMENTS_CODES[self.instruments]
        positions = positions.set_index(['trade_time', 'code'])
        # --- 1. 初始化 ---
        # quality_score 排好序了
        sorted_programs = programs

        selected_program_names = []

        if sorted_programs.empty:
            kd_logger.warning("没有候选策略进入第五关。")
            return pd.DataFrame()

        # --- 2. 序贯筛选循环 ---
        # 逐个考察已排序的候选策略
        for _, p_candidate in sorted_programs.iterrows():
            strategy_name = p_candidate['name']

            # 获取该策略在验证集上的原始表现作为基准
            # 我们使用 val_fitness，因为它是在验证集上计算的原始表现
            perf_original_val = p_candidate['val_fitness']

            # 如果是第一个策略，直接入选 (因为它已经是在quality_score上排名第一的)
            if not selected_program_names:
                kd_logger.info(
                    f"选择 '{strategy_name}' (验证集Perf={perf_original_val:.4f}) 作为初始策略。"
                )
                selected_program_names.append(strategy_name)
                continue

            # 获取当前候选策略和已选策略的验证集仓位
            pos_candidate = positions[strategy_name]
            selected_positions = positions[selected_program_names]

            # --- a. 相关性检验 ---
            max_abs_corr = selected_positions.corrwith(
                pos_candidate, method='spearman').abs().max()
            kd_logger.info(
                f"考察 '{strategy_name}': 与已选组合的最大相关性: {max_abs_corr:.4f}")

            if max_abs_corr > config['corr_threshold']:
                kd_logger.info(
                    f"  - 结果: 剔除 (相关性 > {config['corr_threshold']})")
                continue

            # --- b. 正交化与绩效增益检验 ---
            data_for_regression = pd.concat(
                [pos_candidate.rename('y'), selected_positions],
                axis=1).dropna()
            y = data_for_regression['y']
            X = data_for_regression.drop(columns='y')

            if X.empty or len(X) < 200:
                kd_logger.info(f"  - 结果: 剔除 (回归样本不足)")
                continue

            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            residual_position = pd.Series(y - model.predict(X), index=y.index)

            # 归一化残差仓位
            max_abs_residual = residual_position.abs().max()
            if max_abs_residual > 1e-8:
                normalized_residual_position = residual_position / max_abs_residual
            else:
                normalized_residual_position = residual_position.copy()

            kd_logger.debug(
                f"{strategy_name}  - 残差仓位已归一化，原最大绝对值: {max_abs_residual:.4f}, 现最大绝对值: {normalized_residual_position.abs().max():.4f}"
            )

            # 格式转化
            normalized_residual_position = normalized_residual_position.to_frame(
                name='pos').unstack(level='code')

            # 回测归一化后的残差仓位在验证集上的表现
            df = calculate_ful_ts_ret(pos_data=normalized_residual_position,
                                      total_data=total_data1,
                                      strategy_settings=custom_params)
            residual_fitness = empyrical.sharpe_ratio(returns=df['a_ret'],
                                                      period=empyrical.DAILY)

            retention_ratio = (abs(residual_fitness) /
                               (abs(perf_original_val) + 1e-8))

            #kd_logger.info(f"  - 原始验证集表现: {perf_original_val:.4f}")
            #kd_logger.info(f"  - 残差仓位验证集表现: {residual_fitness:.4f}")
            #kd_logger.info(f"  - 表现保留率: {retention_ratio:.2%}")
            if retention_ratio < config['gain_threshold']:
                kd_logger.info(
                    f"  - 结果: 剔除 (表现保留率 < {config['gain_threshold']:.0%})")
                continue

            # --- c. 入选 ---
            kd_logger.info(f" {strategy_name}  - 结果: ★★★ 入选 ★★★")
            selected_program_names.append(strategy_name)

        # --- 3. 整理最终结果 ---
        final_selected_programs = programs[programs['name'].isin(
            selected_program_names)].sort_values('quality_score',
                                                 ascending=False)  # 按第四关的质量分排序

        kd_logger.info(f"Stage 5后最终精选出 {len(final_selected_programs)} 个策略。")
        return final_selected_programs

    def run(self, programs, train_val_data, val_data, train_val_positions,
            val_positions, custom_params):

        survivors_sc = self.stage_sanity_check(programs=programs,
                                               config=self.config['stage_sc'])

        survivors_oc = self.stage_overfitting_check(
            programs=survivors_sc, config=self.config['stage_oc'])

        survivors_rsc = self.stage_rolling_stability_check(
            programs=survivors_oc,
            positions=train_val_positions,
            total_data=train_val_data,
            custom_params=custom_params,
            config=self.config['stage_rsc'])

        survivors_na = self.stage_neighborhood_analysis(
            programs=survivors_rsc,
            positions=val_positions,
            config=self.config['stage_na'])

        survivors_qs = self.stage_quality_scoring(
            programs=survivors_na, config=self.config['stage_qs'])

        survivors_ss = self.stage_sequential_selection(
            programs=survivors_qs,
            positions=val_positions,
            total_data=val_data,
            custom_params=custom_params,
            config=self.config['stage_ss'])
        return survivors_ss


def prepare_data(instruments, method, task_id):
    pdb.set_trace()
    base_dirs = os.path.join(
        os.path.join('temp', "{}".format(method), str(task_id)))
    programs = load_fitness(base_dirs=base_dirs)
    positions_res = load_positions(base_dirs=base_dirs,
                                   names=programs['name'].tolist())

    val_positions = merge_positions(positions_res=positions_res, mode='val')
    train_positions = merge_positions(positions_res=positions_res,
                                      mode='train')
    val_data = load_data(instruments=instruments, mode='val')
    train_data = load_data(instruments=instruments, mode='train')

    train_val_data = pd.concat([train_data, val_data], axis=0).sort_index()
    train_val_positions = pd.concat([train_positions, val_positions],
                                    axis=0).sort_index()
    return programs, train_val_data, val_data, train_val_positions, val_positions


if __name__ == '__main__':
    method = 'aicso2'
    task_id = '200037'
    instruments = 'ims'

    programs, train_val_data, val_data, train_val_positions, val_positions = prepare_data(
        instruments=instruments, method=method, task_id=task_id)

    config = {
        'stage_sc': {
            # 训练集夏普比率必须大于1.0，保证其至少在样本内找到了有效的模式。
            'min_train_fitness': 1.0,

            # 验证集夏普比率必须大于0.1，确保其在样本外有最基本的生存能力。
            'min_val_fitness': 0.1
        },
        'stage_oc': {
            'min_retention_rate': 0.3  # 训练集到验证集的保留率
        },
        'stage_rsc': {  # 使用120个交易日作为滚动窗口来计算夏普比率
            'rolling_window': 120,

            # 每20个交易日滚动一次窗口
            'rolling_step': 20,

            # 滚动夏普的标准差不能太大，这衡量了表现的平滑度。 正常标准差为1.0, 增加一倍，
            'max_rolling_std': 2.5,

            # 至少有60%的时间窗口内，策略的夏普是正的。
            'min_rolling_win_rate': 0.60,

            # 在表现最差的那个窗口，夏普比率也不能低于-2.0，防止策略有致命缺陷。
            'min_worst_window_perf': -2.3
        },
        'stage_na': {
            'min_cluster_input_size': 10,  # 最小策略数量
            'dbscan_eps':
            0.4,  #两个策略被视为“邻居”的最大行为距离(1 - abs(corr)) # 0.4 意味着仓位序列的斯皮尔曼相关性绝对值 > 0.6 的策略可以互为邻居  0.2 (非常严格) 到 0.5 (比较宽松
            'dbscan_min_samples':
            3,  # 定义了要形成一个“稠密”的簇，一个核心点至少需要多少个邻居  范围建议: 3 到 5。
            'min_cluster_size':
            3,  # 被认为是稳定且有代表性的“行为簇”（家庭），至少需要多个策略 大于等于 dbscan_min_samples。
            'min_cluster_fitness': 0.4,  # 这个簇内所有策略的平均验证集夏普比率，必须达到一个最低标准
            'max_cluster_cv':
            0.8  # 簇内部的绩效稳定性，用变异系数(CV = std/mean)衡量。 这个值越小，说明簇内成员表现越一致，该行为模式对参数变化越不敏感 0.8 意味着标准差不能超过均值的80%，是一个相对合理的容忍度。 范围建议: 0.5 (非常严格) 到 1.0 (比较宽松)。
        },
        'stage_qs': {
            'quality_score_weights': {
                # 验证集表现分排名，给予最高权重   场景一：极度风险厌恶，追求极致的稳定性 0.5 训练集校验集各占一半 0.85, # 绝对表现占绝对主导
                'val_fit': 0.7,

                # 验证集稳定性（低衰减率）排名，给予辅助权重
                'val_stable': 0.3,
                'min_quality_score': 0.2,  ## 最新大大的分数
            }
        },
        'stage_ss': {
            # 对最终入选组合的内部相关性要求更严，比如不超过0.6
            'corr_threshold': 0.6,

            # 要求策略在剔除已有信息后，其独立部分的绩效至少能保留25%
            'gain_threshold': 0.25
        }
    }

    strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]] * 0.05,
        'slippage': 0,  #SLIPPAGE_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }

    custom_params = {'strategy_settings': strategy_settings}

    helper_strategy = HelperStrategy(instruments=instruments,
                                     task_id=task_id,
                                     config=config)
    '''
    result1 = helper_strategy.stage_rolling_stability_check(
        programs=programs,
        positions=train_val_positions,
        total_data=train_val_data,
        custom_params=custom_params,
        config=helper_strategy.config['stage_rsc']
    )
    '''
    filter_strategy = FilterStrategy(instruments=instruments,
                                     task_id=task_id,
                                     config=config)
    survivor = filter_strategy.run(programs=programs,
                                   train_val_data=train_val_data,
                                   val_data=val_data,
                                   train_val_positions=train_val_positions,
                                   val_positions=val_positions,
                                   custom_params=custom_params)
    pdb.set_trace()
    print('-->')
