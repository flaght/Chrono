import os, pdb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from lib.lsx001 import fetch_times
from kdutils.macro2 import *


def train_model(method, task_id, instruments, period):
    random_state = 42
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    filename = os.path.join(dirs, "final_data.feather")
    final_data = pd.read_feather(filename).set_index(['trade_time', 'code'])
    ## 切割训练集 校验集 测试集
    train_data = final_data.loc[
        time_array['train_time'][0]:time_array['val_time'][1]]
    test_data = final_data.loc[
        time_array['test_time'][0]:time_array['test_time'][1]]

    train_data = train_data.dropna()
    test_data = test_data.dropna()
    features = [
        col for col in final_data.columns
        if col not in ['nxt1_ret_{0}h'.format(period)]
    ]
    new_columns = ["f{0}".format(i) for i in range(0, len(features))]

    X = train_data[features]
    X.columns = new_columns
    y = train_data['nxt1_ret_{0}h'.format(period)]

    # 定义交叉验证的折数
    N_SPLITS = 5
    # 【新增】计算每一折验证集的大小
    # n_splits=5 会把数据分成 5+1=6 块，每块作为一次验证集
    test_fold_size = len(X) // (N_SPLITS + 1)

    # 【新增】设置一个合理的最小训练样本数
    # 比如说，我们要求至少要有两块数据的量（即2 * test_fold_size）才开始训练
    # 这是一个可以调整的超参数，你可以从一个保守的值开始
    MIN_TRAIN_SIZE = 2 * test_fold_size

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    models = []
    scalers = []
    val_scores = []
    # 循环遍历每一折数据

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        if len(train_index) < MIN_TRAIN_SIZE:
            print(
                f"--- FOLD {fold + 1}/{N_SPLITS} SKIPPED: train size {len(train_index)} < min_train_size {MIN_TRAIN_SIZE} ---"
            )

            continue  # 跳过当前这一折，进入下一次循环
        print(f"\n=============== FOLD {fold + 1}/{N_SPLITS} ================")
        # 根据索引获取当前折的训练集和验证集
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        print(
            f"训练集大小: {len(X_train)} ({X_train.index.get_level_values('trade_time').min()} to {X_train.index.get_level_values('trade_time').max()})"
        )
        print(
            f"验证集大小: {len(X_val)} ({X_val.index.get_level_values('trade_time').min()} to {X_val.index.get_level_values('trade_time').max()})"
        )

        ### 标准化
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)

        # 将标准化后的 numpy array 转回 DataFrame
        X_train_norm_df = pd.DataFrame(X_train_norm,
                                       index=X_train.index,
                                       columns=X_train.columns)
        X_val_norm_df = pd.DataFrame(X_val_norm,
                                     index=X_val.index,
                                     columns=X_val.columns)

        model = Ridge(alpha=1.0, random_state=random_state)

        # 训练模型
        model.fit(X_train_norm_df, y_train)

        models.append(model)
        scalers.append(scaler)
        y_pred_val = model.predict(X_val_norm_df)
        val_mae = np.mean(np.abs(y_pred_val - y_val))
        val_scores.append(val_mae)

        print(f"本折验证集 MAE: {val_mae:.6f}")

    test_scaled = test_data[features]
    test_scaled.columns = new_columns

    all_predictions = []
    for model, scaler in zip(models, scalers):
        test_scaled1 = scaler.transform(test_scaled)
        prediction = model.predict(test_scaled1)  # 线性模型的 predict 方法更简单
        all_predictions.append(prediction)

    # 取所有模型预测结果的平均值
    raw_meta = np.mean(all_predictions, axis=0)

    predict_data1 = pd.DataFrame(raw_meta,
                                 index=test_data.index,
                                 columns=['predict'])
    predict_data1 = pd.concat(
        [test_data['nxt1_ret_{0}h'.format(period)], predict_data1], axis=1)
    predict_data1.reset_index().to_feather(
        os.path.join(dirs, "rigde_predict_data.feather"))

    print("\n模型训练和预测完成，结果已保存。")
