import os, pdb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

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
    test_fold_size = len(X) // (N_SPLITS + 1)
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
            continue

        print(f"\n=============== FOLD {fold + 1}/{N_SPLITS} ================")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        print(
            f"训练集大小: {len(X_train)} ({X_train.index.get_level_values('trade_time').min()} to {X_train.index.get_level_values('trade_time').max()})"
        )
        print(
            f"验证集大小: {len(X_val)} ({X_val.index.get_level_values('trade_time').min()} to {X_val.index.get_level_values('trade_time').max()})"
        )

        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)

        X_train_norm_df = pd.DataFrame(X_train_norm,
                                       index=X_train.index,
                                       columns=X_train.columns)
        X_val_norm_df = pd.DataFrame(X_val_norm,
                                     index=X_val.index,
                                     columns=X_val.columns)

        # 1. 创建一个用于在当前训练集内部寻找最优alpha的交叉验证器
        #    这对于超参数调优是至关重要的，且必须是时间序列感知的。
        tscv_inner = TimeSeriesSplit(n_splits=3)

        print("正在使用LassoCV在当前训练集上寻找最优alpha...")

        #    - cv=tscv_inner: 告诉LassoCV使用我们定义的时间序列交叉验证方法来评估每个alpha。
        #    - n_jobs=-1: 使用所有可用的CPU核心来并行计算，大大加快alpha的搜索速度。
        #    - max_iter=2000: 增加最大迭代次数，防止在alpha较小时模型不收敛。
        model = LassoCV(cv=tscv_inner,
                        random_state=random_state,
                        n_jobs=-1,
                        max_iter=2000)

        #  训练模型
        #    LassoCV.fit() 会自动在多个alpha候选项上进行训练和评估，并最终使用找到的最佳alpha来重新训练整个X_train_norm_df。
        model.fit(X_train_norm_df, y_train)

        # 4. (非常重要) 打印出LassoCV找到的结果，用于诊断
        print(f"LassoCV 找到的最优 alpha: {model.alpha_:.8f}")
        num_selected_features = np.sum(model.coef_ != 0)
        print(f"保留的特征数量: {num_selected_features} / {len(features)}")

        if num_selected_features == 0:
            print("⚠️ 警告: LassoCV找到的最优alpha仍然导致所有特征系数为0。")

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
        prediction = model.predict(test_scaled1)
        all_predictions.append(prediction)

    raw_meta = np.mean(all_predictions, axis=0)

    predict_data1 = pd.DataFrame(raw_meta,
                                 index=test_data.index,
                                 columns=['predict'])
    predict_data1 = pd.concat(
        [test_data['nxt1_ret_{0}h'.format(period)], predict_data1], axis=1)

    predict_data1.reset_index().to_feather(
        os.path.join(dirs, "lassocv_predict_data.feather"))

    print("\n模型训练和预测完成，结果已保存。")
