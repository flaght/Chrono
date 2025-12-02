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
        
        # --- 结构与复杂度控制 (核心) ---
        # 您的 num_leaves=8, max_depth=4. 这表明一个简单、不易过拟合的模型效果很好。
        # 我们让 max_depth 稍微浮动，并让 num_leaves 在一个相对较小的范围内搜索。
        # 关系: num_leaves <= 2^max_depth
        'max_depth': trial.suggest_int('max_depth', 3, 6), # 中心在4-5，稍微探索更深或更浅
        'num_leaves': trial.suggest_int('num_leaves', 6, 31), # 中心在8附近，最大不超过6层树的32叶
        
        # 您的 min_child_samples=100. 这是一个比较大的值，能有效防止过拟合。
        # 我们以此为中心进行搜索。
        'min_child_samples': trial.suggest_int('min_child_samples', 60, 200),

        # --- 正则化 (防止过拟合的关键) ---
        # 您的 lambda_l1=5.0, lambda_l2=5.0. 这是非常强的正则化，说明模型从强约束中受益。
        # 我们将搜索空间设置在这个强正则化区域。
        'lambda_l1': trial.suggest_float('lambda_l1', 1.0, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 10.0, log=True),
        
        # 您的 min_gain_to_split=0.01. 这也是一个防止过拟合的参数。
        # 我们让它在0附近浮动，可以探索不进行此约束的情况 (0.0)。
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.05),

        # --- 学习过程控制 ---
        # 您的 learning_rate=0.01. 这是一个很好的、较小的学习率。
        # 我们可以探索稍微大一点或小一点的学习率，看看能否加速收敛或找到更优的点。
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        
        # --- 采样 (增加模型多样性) ---
        # 您的 feature_fraction=0.8, bagging_fraction=0.8.
        # 我们以此为中心，探索更大或更小的采样率。
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7), # 您的值为1，我们探索更稀疏的bagging
    }

    # 1.2 执行交叉验证，并收集所有目标的指标
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    sharpe_scores = []
    calmar_scores = []
    ic_mean_scores = []
    avg_ret_scores = []

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

        #  在验证集上运行完整的回测
        val_preds = model.predict(X_val_norm,
                                  num_iteration=model.best_iteration)

        eval_df = pd.DataFrame({
            'trade_time':
            X_val.index.get_level_values('trade_time'),
            'predict':
            val_preds,
            'ret':
            y_val.values
        }).reset_index(drop=True)

        eval_roll_win = min(len(eval_df) // 4, 100)
        if eval_roll_win < 20:
            continue

        eval_df.rename(columns={
            'predict': 'factor'
        },
                       inplace=True)
        evaluator = FactorEvaluate1(factor_data=eval_df,
                                    roll_win=eval_roll_win,
                                    scale_method='raw')
        stats = evaluator.run()

        # 对于无效值，我们赋予一个非常差的默认值，避免优化过程出错
        sharpe_scores.append(stats.get('sharpe2', 0))
        calmar_scores.append(stats.get('calmar', 0))
        ic_mean_scores.append(np.abs(stats.get('ic_mean', 0)))
        avg_ret_scores.append(stats.get('avg_ret', 0))

    # 如果所有折都被跳过，Pruned trial
    if not sharpe_scores:
        raise optuna.exceptions.TrialPruned()

    mean_sharpe = np.mean([s for s in sharpe_scores if np.isfinite(s)])
    mean_calmar = np.mean([c for c in calmar_scores if np.isfinite(c)])
    mean_ic = np.mean([i for i in ic_mean_scores if np.isfinite(i)])
    mean_avg_ret = np.mean([r for r in avg_ret_scores if np.isfinite(r)])

    # Optuna 将会根据 study 的 directions 来最大化或最小化这些值
    return mean_sharpe, mean_calmar, mean_ic, mean_avg_ret


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

    print("\n=============== 开始 Optuna 多目标超参数优化 ================")
    print("优化目标: [Sharpe, Calmar, IC_Mean, Avg_Return]")

    study = optuna.create_study(
        directions=['maximize', 'maximize', 'maximize', 'maximize'],
        study_name=f'lgbm_tuning_{task_id}_{period}')

    study.optimize(
        lambda trial: objective_financial(trial, X, y, random_state, N_SPLITS,
                                          MIN_TRAIN_SIZE),
        n_trials=300  # 多目标优化通常需要更多次试验
    )

    print("\n=============== Optuna 优化完成 ================")

    # 【核心修改】处理多目标优化结果
    # 'best_trials' 包含了所有帕累托最优的试验结果

    pareto_front_trials = study.best_trials
    print(f"找到了 {len(pareto_front_trials)} 个帕累托最优解。")

    print("\n--- 帕累托最优解列表 ---")
    for i, t in enumerate(pareto_front_trials):
        print(f"  解 {i+1} (Trial {t.number}):")
        print(f"    - 参数: {t.params}")
        print(f"    - 指标 (Sharpe, Calmar, IC, AvgRet): {t.values}")

    #  选择一个最终的参数组合
    # 这里的选择策略体现了您的“优先级”
    # 策略: 从所有帕累托最优解中，选择 夏普比率 最高的那个
    if not pareto_front_trials:
        print("警告: 未找到有效的帕累托解，将使用默认参数。")
        best_params = {  # 提供一个安全的回退参数
            'learning_rate': 0.01,
            'num_leaves': 8,
            'max_depth': 4,
            'min_child_samples': 100,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 5.0,
            'lambda_l2': 5.0
        }
    else:
        # t.values[0] 对应我们定义的第一个目标：夏普比率
        best_trial = max(pareto_front_trials, key=lambda t: t.values[0])
        best_params = best_trial.params
        print("\n--- 根据优先级选择最终解 ---")
        print(f"选择策略: 最高的夏普比率 (Sharpe)")
        print(f"最终选择的 Trial: {best_trial.number}")
        print(f"最终选择的参数: {best_params}")
        print(f"最终选择的指标: {best_trial.values}")

    print("\n=============== 使用最终选择的最佳参数训练模型 ================")
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
