import optuna, empyrical
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret
from kdutils.composite.prepare import *


def create_positions1(train_positions, val_positions, test_positions,
                      model_class, params, strategy_settings):
    positions = pd.concat([train_positions, val_positions, test_positions],
                          axis=0)
    positions = positions.set_index('trade_time')


def composite1(train_data,
               val_data,
               test_data,
               train_positions,
               val_positions,
               test_positions,
               instruments,
               names,
               strategy_settings,
               objective_func,
               strategy_func,
               task_name=None):
    train_val_data = pd.concat([train_data, val_data], axis=0)
    train_val_positions = pd.concat([train_positions, val_positions], axis=0)
    total_data_train_val = train_val_data.sort_index().reset_index().copy(
    ).set_index(['trade_time', 'code'])
    train_val_data = create_target1(total_data=train_val_data,
                                    positions=train_val_positions,
                                    instruments=instruments)
    train_val_matrix = train_val_data.set_index('trade_time')[['target'] +
                                                              names]
    y_train_val = train_val_matrix['target']
    X_train_val = train_val_matrix[names]

    study = optuna.create_study(direction='maximize',
                                study_name='robust_synthesis' if
                                not isinstance(task_name, str) else task_name)
    study.optimize(
        lambda trial: objective_func(trial, X_train_val, y_train_val,
                                     total_data_train_val, strategy_settings),
        n_trials=10  # 至少100次
    )

    print("\n--- Optuna交叉验证搜索完成 ---")
    print(f"找到的最佳鲁棒性分数: {study.best_value:.4f}")
    print("对应的参数组合:", study.best_params)
    best_robust_trial = robust_hyperparameters(study,
                                               distance_threshold=0.2,
                                               min_neighbors=3,
                                               minimum_size=7)
    best_params = {
        'object_best': study.best_params,  ## 寻优最佳
        'robust_best': best_robust_trial.params  ## 邻里最佳
    }
    return best_params


### 针对通用方式
def objective_cv1(X_train_val: pd.DataFrame, y_train_val: pd.Series,
                  total_data_train_val: pd.DataFrame, strategy_settings: dict,
                  model_class: any, params: dict):
    # --- 2. 使用TimeSeriesSplit进行交叉验证 ---
    # (这部分逻辑与LGBM版本完全相同)
    tscv = TimeSeriesSplit(n_splits=5)
    fold_sharpes = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
        X_train_fold, X_val_fold = X_train_val.iloc[
            train_idx], X_train_val.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_val.iloc[
            train_idx], y_train_val.iloc[val_idx]

        try:
            model = model_class(**params)
            model.fit(X_train_fold, y_train_fold)

            # 在当前折叠的验证集上预测
            predicted_labels = model.predict(X_val_fold)

            # 回测并计算夏普
            position_map = {0: -1, 1: 0, 2: 1}
            val_positions = pd.Series(predicted_labels,
                                      index=X_val_fold.index).map(position_map)

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
            sharpe = empyrical.sharpe_ratio(pnl_df['a_ret'], period='daily')
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


##针对lightBGM train 方法
def objective_cv2(X_train_val: pd.DataFrame, y_train_val: pd.Series,
                  total_data_train_val: pd.DataFrame, strategy_settings: dict,
                  params: dict):
    # --- 2. 使用TimeSeriesSplit进行交叉验证 ---
    # (这部分逻辑与LGBM版本完全相同)
    pdb.set_trace()
    tscv = TimeSeriesSplit(n_splits=5)
    fold_sharpes = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
        X_train_fold, X_val_fold = X_train_val.iloc[
            train_idx], X_train_val.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_val.iloc[
            train_idx], y_train_val.iloc[val_idx]
        lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
        lgb_val = lgb.Dataset(X_val_fold, label=y_val_fold)

        try:
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_val],
                valid_names=['val'],
                callbacks=[lgb.early_stopping(30, verbose=False)])

            # 在当前折叠的验证集上预测和回测
            predicted_probs = model.predict(X_val_fold,
                                            num_iteration=model.best_iteration)
            predicted_labels = np.argmax(predicted_probs, axis=1)
            position_map = {0: -1, 1: 0, 2: 1}
            val_positions = pd.Series(predicted_labels,
                                      index=X_val_fold.index).map(position_map)

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
            sharpe = empyrical.sharpe_ratio(pnl_df['a_ret'], period='daily')
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


def robust_hyperparameters(study: optuna.Study,
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
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
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
        neighbor_indices = param_distance_matrix.index[param_distance_matrix[i]
                                                       < distance_threshold]

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


def objective_rf_cv(trial: optuna.Trial, X_train_val: pd.DataFrame,
                    y_train_val: pd.Series, total_data_train_val: pd.DataFrame,
                    strategy_settings: dict) -> float:
    # --- 1. 定义超参数搜索空间 ---
    params = {
        # 树的数量
        'n_estimators':
        trial.suggest_int('n_estimators', 100, 500, step=50),

        # 控制单棵树的复杂度，防止过拟合
        'max_depth':
        trial.suggest_int('max_depth', 5, 20),
        'min_samples_leaf':
        trial.suggest_int('min_samples_leaf', 20, 200),
        'min_samples_split':
        trial.suggest_int('min_samples_split', 40, 400),

        # 控制随机性
        'max_features':
        trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8]),

        # 其他参数
        'n_jobs':
        -1,
        'random_state':
        42,
        'class_weight':
        'balanced'
    }

    return objective_cv1(X_train_val=X_train_val,
                         y_train_val=y_train_val,
                         total_data_train_val=total_data_train_val,
                         strategy_settings=strategy_settings,
                         model_class=RandomForestClassifier,
                         params=params)


def objective_lbgm_csv(trial: optuna.Trial, X_train_val: pd.DataFrame,
                       y_train_val: pd.Series,
                       total_data_train_val: pd.DataFrame,
                       strategy_settings: dict):

    # 1. 定义超参数搜索空间 (包含DART和强力正则化)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbose': -1,
        'n_jobs': -1,
        'device': 'gpu',
        'seed': 42,
        #'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'learning_rate': trial.suggest_float('learning_rate',
                                             0.01,
                                             0.1,
                                             log=True),
        'num_leaves': trial.suggest_int('num_leaves', 7, 40),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 250),
    }
    return objective_cv2(X_train_val=X_train_val,
                         y_train_val=y_train_val,
                         total_data_train_val=total_data_train_val,
                         strategy_settings=strategy_settings,
                         params=params)


def rf_classifer_cv(train_data,
                    val_data,
                    test_data,
                    train_positions,
                    val_positions,
                    test_positions,
                    instruments,
                    names,
                    strategy_settings,
                    key=None):
    pdb.set_trace()
    composite1(train_data=train_data,
               val_data=val_data,
               test_data=test_data,
               train_positions=train_positions,
               val_positions=val_positions,
               test_positions=test_positions,
               instruments=instruments,
               names=names,
               strategy_settings=strategy_settings,
               objective_func=objective_rf_cv,
               task_name='RandomForestClassifer')


def lbgm_classifer_cv(train_data,
                      val_data,
                      test_data,
                      train_positions,
                      val_positions,
                      test_positions,
                      instruments,
                      names,
                      strategy_settings,
                      key=None):
    composite1(train_data=train_data,
               val_data=val_data,
               test_data=test_data,
               train_positions=train_positions,
               val_positions=val_positions,
               test_positions=test_positions,
               instruments=instruments,
               names=names,
               strategy_settings=strategy_settings,
               objective_func=objective_lbgm_csv,
               task_name='LightBGMClassifer')
