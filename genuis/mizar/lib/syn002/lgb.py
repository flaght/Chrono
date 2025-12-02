import os, pdb
import optuna
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from lib.lsx001 import fetch_times
from lib.cux001 import FactorEvaluate1

from kdutils.macro2 import *


def objective_financial(trial, X, y, random_state, N_SPLITS, MIN_TRAIN_SIZE):
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 1000,
        'verbose': -1,
        'n_jobs': -1,
        'boosting_type': 'gbdt',
        # --- 以下是 Optuna 优化的超参数 ---
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 6, 24),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 250),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.5, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.5, 10.0, log=True),
    }
    # 执行交叉验证
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    val_icirs = []  # 存储每折的 ICIR

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        if len(train_index) < MIN_TRAIN_SIZE:
            continue

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)

        lgb_train = lgb.Dataset(pd.DataFrame(X_train_norm, columns=X.columns),
                                y_train)
        lgb_val = lgb.Dataset(pd.DataFrame(X_val_norm, columns=X.columns),
                              y_val,
                              reference=lgb_train)

        params['seed'] = random_state + fold

        model = lgb.train(params,
                          lgb_train,
                          valid_sets=[lgb_val],
                          callbacks=[lgb.early_stopping(50, verbose=False)])

        # 【核心】在验证集上进行迷你回测
        val_preds = model.predict(X_val_norm,
                                  num_iteration=model.best_iteration)

        # 构造 FactorEvaluate1 需要的 DataFrame
        eval_df = pd.DataFrame({
            'trade_time':
            X_val.index.get_level_values('trade_time'),
            'factor':
            val_preds,
            'ret':
            y_val.values
        }).reset_index(drop=True)

        # 为验证集上的评估设置一个合理的滚动窗口
        eval_roll_win = min(len(eval_df) // 4, 100)  # 例如，验证集大小的1/4，最多100
        if eval_roll_win < 20:  # 确保窗口不要太小，否则IC计算不稳定
            continue

        evaluator = FactorEvaluate1(
            factor_data=eval_df,
            factor_name='factor',
            ret_name='ret',
            roll_win=eval_roll_win,
            scale_method='raw'  # 模型输出的预测值直接作为因子，不再缩放
        )

        stats = evaluator.run()

        ic_ir = stats.get('ic_ir', 0)

        # 确保结果是有效的数字
        if np.isfinite(ic_ir):
            val_icirs.append(ic_ir)

    # 3.4 返回优化目标
    if not val_icirs:
        return float('inf')  # 如果没有有效的评估结果，返回一个很差的值，让Optuna放弃这个方向
    pdb.set_trace()
    mean_icir = np.mean(val_icirs)
    print(f"Trial {trial.number}: Mean ICIR = {mean_icir:.4f}")

    # Optuna 默认是最小化，所以我们需要最大化 ICIR，即最小化 -ICIR
    return -mean_icir


def objective_with_penalty(trial, X, y, random_state, N_SPLITS,
                           MIN_TRAIN_SIZE):
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 1000,  # n_estimators 保持较大，通过 early_stopping 控制
        'verbose': -1,
        'n_jobs': -1,
        'boosting_type': 'gbdt',
        # --- 以下是 Optuna 优化的超参数 ---
        'learning_rate': trial.suggest_float('learning_rate',
                                             0.01,
                                             0.1,
                                             log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 64),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-2, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-2, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0,
                                                 0.1),
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    val_scores = []
    prediction_volatilities = []  # 【新增】用于存储每折预测的波动性

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        if len(train_index) < MIN_TRAIN_SIZE:
            continue

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        ### 标准化
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)

        X_train_norm_df = pd.DataFrame(X_train_norm,
                                       index=X_train.index,
                                       columns=X_train.columns)
        X_val_norm_df = pd.DataFrame(X_val_norm,
                                     index=X_val.index,
                                     columns=X_val.columns)

        lgb_train = lgb.Dataset(X_train_norm_df, y_train)
        lgb_val = lgb.Dataset(X_val_norm_df, y_val, reference=lgb_train)

        # 为每个 fold 设置不同的随机种子
        params['seed'] = random_state + fold

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(50, verbose=False)
                       ]  # 在优化时不打印 early_stopping 日志
        )

        # 【新增】获取验证集预测并计算其标准差
        val_preds = model.predict(X_val_norm,
                                  num_iteration=model.best_iteration)
        prediction_volatilities.append(np.std(val_preds))

        metric_name = list(model.best_score['val'].keys())[0]
        val_scores.append(model.best_score['val'][metric_name])

    if not val_scores:
        # 如果因为训练集太小所有折都被跳过，返回一个很大的值
        return float('inf')

    mean_mae = np.mean(val_scores)
    mean_volatility = np.mean(prediction_volatilities)

    # 【核心修改】定义惩罚权重。这个权重是一个超参数，需要微调。
    # 我们可以从一个较小的值开始，比如 0.1。
    # 这意味着我们愿意接受 MAE 稍微差一点，来换取更稳定的预测。
    penalty_weight = trial.suggest_float('penalty_weight', 0.05,
                                         0.5)  # 也可以让Optuna自己找

    # 最终的目标函数值 = 平均MAE + 惩罚项
    final_score = mean_mae + penalty_weight * mean_volatility

    # 打印中间信息，方便观察
    print(
        f"Trial {trial.number}: MAE={mean_mae:.6f}, Volatility={mean_volatility:.6f}, Final Score={final_score:.6f}"
    )

    return final_score


def optuna_model(method, task_id, instruments, period):
    random_state = 42
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    filename = os.path.join(dirs, "final_data.feather")
    final_data = pd.read_feather(filename).set_index(['trade_time', 'code'])

    train_data = final_data.loc[
        time_array['train_time'][0]:time_array['val_time'][1]]
    test_data = final_data.loc[
        time_array['test_time'][0]:time_array['test_time'][1]]

    ## 切割训练集 校验集 测试集
    train_data = final_data.loc[
        time_array['train_time'][0]:time_array['val_time'][1]]
    test_data = final_data.loc[
        time_array['test_time'][0]:time_array['test_time'][1]]

    train_data = train_data.dropna()
    test_data = test_data.dropna()
    features = [
        col for col in final_data.columns
        if col not in [f'nxt1_ret_{period}h']
    ]
    new_columns = [f"f{i}" for i in range(len(features))]

    X = train_data[features]
    X.columns = new_columns
    y = train_data[f'nxt1_ret_{period}h']

    # 定义交叉验证的折数和最小训练样本数
    N_SPLITS = 5
    test_fold_size = len(X) // (N_SPLITS + 1)
    MIN_TRAIN_SIZE = 2 * test_fold_size

    study = optuna.create_study(direction='minimize',
                                study_name=f'lgbm_tuning_{task_id}_{period}')

    print("=============== 开始 Optuna 超参数优化 ================")
    study.optimize(lambda trial: objective_financial(trial, X, y, random_state,
                                                     N_SPLITS, MIN_TRAIN_SIZE),
                   n_trials=50)

    print("\n=============== Optuna 超参数优化完成 ================")
    print(f"优化的总次数: {len(study.trials)}")
    print(f"找到的最佳 MAE: {study.best_value:.6f}")
    print("找到的最佳超参数:")
    print(study.best_params)

    print("\n=============== 使用最佳参数训练最终模型 ================")
    best_params = study.best_params

    # 将一些固定参数加回到 best_params 中
    best_params['objective'] = 'regression_l1'
    best_params['metric'] = 'mae'
    best_params['n_estimators'] = 1000  # 同样使用 early_stopping
    best_params['verbose'] = -1
    best_params['n_jobs'] = -1
    best_params['boosting_type'] = 'gbdt'

    # 使用与优化时相同的交叉验证设置来训练一组最终模型
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    models = []
    scalers = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        if len(train_index) < MIN_TRAIN_SIZE:
            continue
        print(f"\n--- 训练最终模型: FOLD {fold + 1}/{N_SPLITS} ---")

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)

        X_train_norm_df = pd.DataFrame(X_train_norm,
                                       index=X_train.index,
                                       columns=X_train.columns)
        X_val_norm_df = pd.DataFrame(X_val_norm,
                                     index=X_val.index,
                                     columns=X_val.columns)

        lgb_train = lgb.Dataset(X_train_norm_df, y_train)
        lgb_val = lgb.Dataset(X_val_norm_df, y_val, reference=lgb_train)

        best_params['seed'] = random_state + fold

        model = lgb.train(best_params,
                          lgb_train,
                          valid_sets=[lgb_train, lgb_val],
                          valid_names=['train', 'val'],
                          callbacks=[lgb.early_stopping(50, verbose=True)])
        models.append(model)
        scalers.append(scaler)

    print("\n=============== 在测试集上进行预测 ================")
    X_test = test_data[features]
    X_test.columns = new_columns

    all_predictions = []
    for model, scaler in zip(models, scalers):
        X_test_scaled = scaler.transform(X_test.copy())
        prediction = model.predict(X_test_scaled,
                                   num_iteration=model.best_iteration)
        all_predictions.append(prediction)

    raw_meta = np.mean(all_predictions, axis=0)

    predict_data1 = pd.DataFrame(raw_meta,
                                 index=test_data.index,
                                 columns=['predict'])
    predict_data1 = pd.concat(
        [test_data[f'nxt1_ret_{period}h'], predict_data1], axis=1)
    predict_data1.reset_index().to_feather(
        os.path.join(dirs, "lgbm_predict_data.feather"))

    print("\n模型训练和预测完成，结果已保存。")
