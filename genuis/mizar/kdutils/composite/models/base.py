import optuna, empyrical, pdb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.spatial.distance import pdist, squareform
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret


class Base(object):

    def __init__(self, name):
        self.name = name
        self.y_map = {-1: 0, 0: 1, 1: 2}

    ## y值处理
    def create_target1(self,
                       total_data,
                       positions,
                       price_col='close',
                       neutral_threshold=0.00023 * 0.05):
        data = total_data.sort_values(
            by=['code', 'trade_time']).copy().reset_index()
        data = data[['trade_time', 'code',
                     price_col]].set_index(['trade_time', 'code'])[price_col]
        data = data.unstack()
        future_log_return = np.log((data / data.shift(1))).shift(-1)
        y_target = future_log_return.dropna()  #np.sign(future_log_return)

        y_target = y_target.reset_index().rename(columns={'IM': 'target'})

        y_target = y_target.set_index('trade_time')['target']
        y = pd.Series(0, index=y_target.index, name='target')
        y[y_target > neutral_threshold] = 1
        y[y_target < -neutral_threshold] = -1
        y_target = y
        return positions.merge(y_target, on=['trade_time'])

    def prepare(self, train_data, val_data, train_positions, val_positions,
                neutral_threshold):
        train_val_data = pd.concat([train_data, val_data], axis=0)
        train_val_positions = pd.concat([train_positions, val_positions],
                                        axis=0)
        total_data_train_val = train_val_data.sort_index().reset_index().copy(
        ).set_index(['trade_time', 'code'])
        train_val_data = self.create_target1(
            total_data=train_val_data,
            positions=train_val_positions,
            neutral_threshold=neutral_threshold)

        train_val_matrix = train_val_data.set_index('trade_time')[['target'] +
                                                                  self.names]
        y_train_val = train_val_matrix['target']
        X_train_val = train_val_matrix[self.names]
        return X_train_val, y_train_val, total_data_train_val

    ## 参数确定后训练准备数据
    def prepare1(self, train_data, val_data, train_positions, val_positions,
                 neutral_threshold):
        X_train_val, y_train_val, total_data_train_val = self.prepare(
            train_data=train_data,
            val_data=val_data,
            train_positions=train_positions,
            val_positions=val_positions,
            neutral_threshold=neutral_threshold)
        return X_train_val, y_train_val

    ## 参数确定后训练准备数据
    def prepare2(self, train_data, val_data, train_positions, val_positions,
                 neutral_threshold):
        train_data = self.create_target1(total_data=train_data,
                                         positions=train_positions,
                                         neutral_threshold=neutral_threshold)

        val_data = self.create_target1(total_data=val_data,
                                       positions=val_positions,
                                       neutral_threshold=neutral_threshold)

        train_matrix = train_data.set_index('trade_time')[['target'] +
                                                          self.names]
        y_train = train_matrix['target']
        X_train = train_matrix[self.names]

        val_matrix = val_data.set_index('trade_time')[['target'] + self.names]
        y_val = val_matrix['target']
        X_val = val_matrix[self.names]

        return X_train, y_train, X_val, y_val

    def objective_cv1(self, X_train_val: pd.DataFrame, y_train_val: pd.Series,
                      total_data_train_val: pd.DataFrame,
                      strategy_settings: dict, params: dict):
        # --- 2. 使用TimeSeriesSplit进行交叉验证 ---
        tscv = TimeSeriesSplit(n_splits=5)
        fold_sharpes = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
            X_train_fold, X_val_fold = X_train_val.iloc[
                train_idx], X_train_val.iloc[val_idx]
            y_train_fold, y_val_fold = y_train_val.iloc[
                train_idx], y_train_val.iloc[val_idx]

            try:
                model = self._fit(X_train_fold=X_train_fold,
                                  y_train_fold=y_train_fold,
                                  X_val_fold=X_val_fold,
                                  y_val_fold=y_val_fold,
                                  params=params)

                # 在当前折叠的验证集上预测
                predicted_labels = self._predict(model=model,
                                                 X_fold=X_val_fold)

                val_positions = pd.Series(predicted_labels,
                                          index=X_val_fold.index)

                val_positions = self.normalize_target(val_positions)

                # 回测并计算夏普
                val_positions.name = 'pos'
                val_positions = val_positions.reset_index()
                val_positions['code'] = 'IM'
                val_positions = val_positions.set_index(['trade_time',
                                                         'code']).unstack()

                # 回测并计算夏普
                pnl_df = calculate_ful_ts_ret(
                    pos_data=val_positions,
                    total_data=total_data_train_val.unstack(),
                    strategy_settings=strategy_settings)
                sharpe = empyrical.sharpe_ratio(pnl_df['a_ret'],
                                                period='daily')
                fold_sharpes.append(sharpe if not np.isnan(sharpe) else 0)

            except Exception as e:
                print(f"RF Fold {fold+1} 训练/评估失败: {e}")
                fold_sharpes.append(-np.inf)

        # --- 3. 计算最终的鲁棒性分数 ---
        if not fold_sharpes:
            return -np.inf

        mean_perf = np.mean(fold_sharpes)
        std_perf = np.std(fold_sharpes)

        # 奖励高均值，惩罚高波动
        robustness_score = mean_perf - 0.5 * std_perf
        return robustness_score

    def robust_hyperparameters(self,
                               study: optuna.Study,
                               distance_threshold: float = 0.15,
                               min_neighbors: int = 3,
                               minimum_size: int = 20):
        """
        对Optuna的优化结果进行事后的邻里审查，找到最稳健的参数组合。
        能够处理数值型和类别型混合的超参数。

        :param study: Optuna的study对象。
        :param distance_threshold: 定义“邻居”的最大参数空间距离。
        :param min_neighbors: 一个点被认为是稳定点所需要的最小邻居数量。
        :return: Optuna.trial.Trial, 最稳健的试验。
        """
        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed_trials) < minimum_size:  # 试验次数太少，审查无意义
            print("试验次数过少，直接返回最佳值试验。")
            return study.best_trial

        params_list = [t.params for t in completed_trials]
        values = np.array([t.value for t in completed_trials])
        params_df = pd.DataFrame(params_list)

        numeric_features = params_df.select_dtypes(
            include=np.number).columns.tolist()
        categorical_features = params_df.select_dtypes(
            exclude=np.number).columns.tolist()

        # ===== 关键修正：统一类别特征的数据类型为字符串 =====
        # 在进行任何处理之前，先把所有类别列都变成 str 类型
        for col in categorical_features:
            params_df[col] = params_df[col].astype(str)

        print("识别出的数值型参数:", numeric_features)
        print("识别出的类别型参数:", categorical_features)

        # 2. 创建一个ColumnTransformer进行预处理
        #    对数值型用MinMaxScaler，对类别型用OneHotEncoder
        preprocessor = ColumnTransformer(transformers=[
            ('num', MinMaxScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore',
                                  sparse_output=False), categorical_features)
        ],
                                         remainder='passthrough')

        # 3. 对参数DataFrame进行拟合和转换
        # 这里的 fit_transform 现在不会再报错了
        params_scaled = preprocessor.fit_transform(params_df)

        # 4. 计算参数间的距离矩阵
        # 现在 params_scaled 是一个纯数值的NumPy数组
        param_distance_matrix = pd.DataFrame(squareform(pdist(params_scaled)))

        # 5. 为每个试验计算鲁棒性分数
        robustness_scores = []
        for i in range(len(completed_trials)):
            # 找到参数空间中距离小于阈值的邻居
            neighbor_indices = param_distance_matrix.index[
                param_distance_matrix[i] < distance_threshold]

            if len(neighbor_indices) < min_neighbors:
                robustness_scores.append(-np.inf)  # 邻居太少，不稳定
                continue

            # 计算邻域的统计特性
            neighbor_values = values[neighbor_indices]
            mean_perf = np.mean(neighbor_values)
            std_perf = np.std(neighbor_values)

            # 鲁棒性分数 = 邻域平均表现 - 邻域表现的波动性
            # 这里的权重 '1.0' 是风险厌恶系数，可以调整
            score = mean_perf - 1.0 * std_perf
            robustness_scores.append(score)

        # 找到那个拥有“最佳邻域”的试验
        best_robust_idx = np.argmax(robustness_scores)
        best_robust_trial = completed_trials[best_robust_idx]

        print("\n--- 邻里审查完成 ---")
        print(
            f"原始最佳试验 (by value): {study.best_trial.number}, value={study.best_trial.value:.4f}"
        )
        print(
            f"最稳健试验 (by neighborhood score): {best_robust_trial.number}, value={best_robust_trial.value:.4f}"
        )
        print("找到的最稳健参数组合:", best_robust_trial.params)

        return best_robust_trial

    ## 参数寻优评估
    def optuna(self, train_data, val_data, train_positions, val_positions,
               objective_func):
        study = optuna.create_study(
            direction='maximize',
            study_name='robust_synthesis'
            if not isinstance(self.task_name, str) else self.task_name)
        X_train_val, y_train_val, total_data_train_val = self.prepare(
            train_data=train_data,
            val_data=val_data,
            train_positions=train_positions,
            val_positions=val_positions,
            neutral_threshold=self.neutral_threshold)
        study.optimize(
            lambda trial:
            objective_func(trial, X_train_val, y_train_val,
                           total_data_train_val, self.strategy_settings),
            n_trials=10  # 至少100次
        )
        pdb.set_trace()
        print("\n--- Optuna交叉验证搜索完成 ---")
        print(f"找到的最佳鲁棒性分数: {study.best_value:.4f}")
        print("对应的参数组合:", study.best_params)
        best_robust_trial = self.robust_hyperparameters(study,
                                                        distance_threshold=0.2,
                                                        min_neighbors=3,
                                                        minimum_size=7)
        best_params = {
            'object_best': study.best_params,  ## 寻优最佳
            'robust_best': best_robust_trial.params  ## 邻里最佳
        }
        return best_params

    ### 入口函数
    def run(self, train_data, val_data, train_positions, val_positions,
            test_positions):
        res = []
        best_params = self.optuna(train_data=train_data,
                                  val_data=val_data,
                                  train_positions=train_positions,
                                  val_positions=val_positions)
        for k in best_params:
            model = self.train(train_data=train_data,
                               val_data=val_data,
                               train_positions=train_positions,
                               val_positions=val_positions,
                               params=best_params[k])
            raw_meta_signal = self.predict(model=model,
                                           train_positions=train_positions,
                                           val_positions=val_positions,
                                           test_positions=test_positions,
                                           name=k)
            res.append(raw_meta_signal)
        return res
