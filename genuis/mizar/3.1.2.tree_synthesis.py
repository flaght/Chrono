import pandas as pd
import numpy as np
import pdb, argparse
import os, pdb, math, itertools
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from dotenv import load_dotenv

load_dotenv()
from ultron.factor.genetic.geneticist.operators import *
from lib.lsx001 import fetch_data1, create_factors, fetch_chosen_factors, fetch_times
from kdutils.macro2 import *


def build_factors(method,
                  instruments,
                  period,
                  datasets=['train', 'val', 'test']):
    expressions = fetch_chosen_factors(method=method, instruments=instruments)
    total_data = fetch_data1(method=method,
                             instruments=instruments,
                             datasets=datasets,
                             period=period,
                             expressions=expressions)
    factors_data = create_factors(total_data=total_data,
                                  expressions=expressions)
    dirs = os.path.join(base_path, method, instruments, 'temp', "tree",
                        str(period))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs, "final_data.feather")
    final_data = factors_data.reset_index().merge(
        total_data[['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)]],
        on=['trade_time', 'code'])
    final_data.to_feather(filename)


def train_model(method, instruments, period):
    random_state = 42
    time_array = fetch_times(method=method, instruments=instruments)
    dirs = os.path.join(base_path, method, instruments, 'temp', "tree",
                        str(period))
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
    pdb.set_trace()
    new_columns = ["f{0}".format(i) for i in range(0, len(features))]
    X_train_scaled = train_data[features]
    X_train_scaled.columns = new_columns
    y_transformed = train_data['nxt1_ret_{0}h'.format(period)]

    X_train, X_val, y_train, y_val = train_test_split(X_train_scaled,
                                                      y_transformed,
                                                      test_size=0.2,
                                                      shuffle=False)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
    params = {
        'objective': 'regression_l1',  # 明确目标是优化 MAE
        'metric': 'mae',
        'n_estimators': 1000,  # 增加迭代次数，给早停更多空间
        'learning_rate': 0.01,  # 降低学习率，让学习过程更平滑
        'feature_fraction': 0.8,  # 每次迭代随机选择80%的特征
        'bagging_fraction': 0.8,  # 每次迭代随机选择80%的数据
        'bagging_freq': 1,
        'lambda_l1': 0.1,  # L1 正则化，有助于特征选择
        'lambda_l2': 0.1,  # L2 正则化，防止权重过大
        'num_leaves': 16,  # 从一个较小的值开始，限制树的复杂度
        'min_child_samples': 20,  # 提高叶子节点的最小样本数，防止过拟合
        'verbose': 1,  # 设为 -1 避免冗余信息
        'n_jobs': -1,
        'seed': random_state,
        'boosting_type': 'gbdt',
    }
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(10, verbose=True)]  # 早停是关键  
    )

    feature_importances = pd.Series(model.feature_importance(), index=features)
    test_scaled = test_data[features]
    test_scaled.columns = new_columns
    raw_meta = model.predict(test_scaled[new_columns], num_iteration=model.best_iteration)
    pdb.set_trace()
    predict_data1 = pd.DataFrame(raw_meta,index=test_data.index, columns=['predict'])
    predict_data1 = pd.concat([test_data['nxt1_ret_{0}h'.format(period)], predict_data1],axis=1)
    predict_data1.reset_index().to_feather(os.path.join(dirs, "predict_data.feather"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--method',
                        type=str,
                        default='cicso0',
                        help='data method')
    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instruments')

    parser.add_argument('--period', type=int, default=5, help='period')

    args = parser.parse_args()
    #build_factors(method=args.method, instruments=args.instruments, period=args.period)
    train_model(method=args.method,
                instruments=args.instruments,
                period=args.period)
