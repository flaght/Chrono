import os, pdb, sys, json, math, empyrical
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split


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


def merge_positions(positions_res, mode):
    res = []
    for name in positions_res:
        positions = positions_res[name][mode]
        positions = positions.rename(columns={'pos': name})
        res.append(positions.set_index('trade_time'))
    positions = pd.concat(res, axis=1).reset_index()
    return positions


## 加载总览fitness
def load_fitness(base_dirs):
    fitness_file = os.path.join(base_dirs, "fitness.feather")
    fitness_pd = pd.read_feather(fitness_file)

    return fitness_pd


def load_data(mode='train'):
    filename = os.path.join(base_path, method, g_instruments, 'level2',
                            '{0}_data.feather'.format(mode))
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    return factors_data.set_index('trade_time')


def create_target1(total_data, positions, price_col='close'):
    data = total_data.sort_values(
        by=['code', 'trade_time']).copy().reset_index()
    data = data[['trade_time', 'code',
                 price_col]].set_index(['trade_time', 'code'])[price_col]
    data = data.unstack()
    future_log_return = np.log((data / data.shift(1))).shift(-1)
    y_target = np.sign(future_log_return)
    y_target = y_target.dropna().astype(int)
    y_target = y_target.reset_index().rename(
        columns={INSTRUMENTS_CODES[g_instruments]: 'target'})
    return positions.merge(y_target, on=['trade_time'])


def create_target2(total_data, positions, price_col='close'):
    data = total_data.sort_values(
        by=['code', 'trade_time']).copy().reset_index()
    data = data[['trade_time', 'code',
                 price_col]].set_index(['trade_time', 'code'])[price_col]
    data = data.unstack()
    future_log_return = np.log((data / data.shift(1))).shift(-1)
    y_target = future_log_return.dropna()  #np.sign(future_log_return)
    y_target = y_target.reset_index().rename(
        columns={INSTRUMENTS_CODES[g_instruments]: 'target'})
    return positions.merge(y_target, on=['trade_time'])


def rank_normalize_signal(raw_signal: pd.Series) -> pd.Series:
    """通过排序并映射到[-1, 1]区间来进行标准化。"""
    ranked_signal = raw_signal.rank(pct=True)
    normalized_signal = (ranked_signal * 2) - 1
    return normalized_signal


def rigde_regression_synthesis(train_data, val_data, test_data,
                               train_positions, val_positions, test_positions,
                               names):
    random_state = 42
    train_data = create_target2(total_data=train_data,
                                positions=train_positions)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data[names])
    y_train = train_data['target']
    pdb.set_trace()
    # --- 3. 创建并训练 RidgeCV 模型 ---
    print("正在使用RidgeCV寻找最佳alpha并训练模型...")
    '''
    time_series_cv = KFold(n_splits=5, shuffle=False)
    
    # 创建RidgeCV实例
    model = RidgeCV(
        alphas=alphas_to_try,
        cv=time_series_cv,
        scoring='neg_mean_squared_error', # 使用负MSE作为评分标准
        store_cv_values=True # 可以存储交叉验证的结果以便分析
    )
    '''

    # 定义一系列候选的alpha值，RidgeCV会从中选择
    alphas_to_try = np.logspace(-3, 3, 10)  # 例如，从 0.001 到 1000 尝试10个值
    model = RidgeCV(
        alphas=alphas_to_try,
        cv=5,
        scoring='neg_mean_squared_error',  # 使用负MSE作为评分标准
        store_cv_values=False)
    model.fit(X_train_scaled, y_train)

    ## 合并所有的仓位数据
    positions = pd.concat([train_positions, val_positions, test_positions],
                          axis=0)
    positions = positions.set_index('trade_time')
    X_scaled = scaler.transform(positions[names])
    raw_meta_signal = model.predict(X_scaled)
    raw_meta_signal = pd.Series(raw_meta_signal, index=positions.index)
    # --- 6. 对输出信号进行标准化，使其成为可交易的仓位 ---
    meta_positions = rank_normalize_signal(raw_meta_signal)
    meta_positions = pd.Series(meta_positions, index=positions.index)
    meta_positions.name = "rigde"
    return meta_positions


def lasso_regression_synthesis(train_data, val_data, test_data,
                               train_positions, val_positions, test_positions,
                               names):
    random_state = 42
    train_data = create_target2(total_data=train_data,
                                positions=train_positions)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data[names])
    y_train = train_data['target']
    pdb.set_trace()
    ## 此处可以考虑是否加入kfold 增强模型稳定性
    model = LassoCV(eps=1e-4,
                    n_alphas=100,
                    cv=5,
                    random_state=random_state,
                    n_jobs=-1)

    model.fit(X_train_scaled, y_train)

    ## 合并所有的仓位数据
    positions = pd.concat([train_positions, val_positions, test_positions],
                          axis=0)
    positions = positions.set_index('trade_time')
    X_scaled = scaler.transform(positions[names])
    raw_meta_signal = model.predict(X_scaled)
    raw_meta_signal = pd.Series(raw_meta_signal, index=positions.index)
    # --- 6. 对输出信号进行标准化，使其成为可交易的仓位 ---
    meta_positions = rank_normalize_signal(raw_meta_signal)
    meta_positions = pd.Series(meta_positions, index=positions.index)
    meta_positions.name = "lasso"
    return meta_positions


def rank_transform(series: pd.Series) -> pd.Series:
    ranked = series.rank(pct=True)
    return (ranked * 2) - 1


def random_forest_synthesis(train_data, val_data, test_data, train_positions,
                            val_positions, test_positions, names):
    train_data = create_target2(total_data=train_data,
                                positions=train_positions)
    random_state = 42
    #X_train_scaled = scaler.fit_transform(train_data[names])
    X = train_data[names]
    y_transformed = rank_transform(train_data['target'])
    pdb.set_trace()
    model = RandomForestRegressor(
        n_estimators=200,  # 树的数量
        max_depth=10,  # 限制每棵树的最大深度，防止过拟合
        min_samples_leaf=50,  # 叶子节点的最小样本数，重要的正则化参数
        max_features='sqrt',  # 每次分裂只考虑 sqrt(n_features) 个特征
        n_jobs=-1,  # 使用所有CPU核心
        random_state=random_state,
        oob_score=True  # 使用袋外样本进行验证，是一个很好的内置验证机制
    )
    model.fit(X, y_transformed)
    pdb.set_trace()
    print(f"模型训练完成。袋外OOB Score (R-squared): {model.oob_score_:.4f}")
    feature_importances = pd.Series(model.feature_importances_,
                                    index=X.columns)
    positions = pd.concat([train_positions, val_positions, test_positions],
                          axis=0)
    positions = positions.set_index('trade_time')
    raw_meta_signal = model.predict(positions[names])
    raw_meta_signal = pd.Series(raw_meta_signal, index=positions.index)
    meta_positions = rank_normalize_signal(raw_meta_signal)

    meta_positions = pd.Series(meta_positions, index=positions.index)
    meta_positions.name = "random_forest"
    return meta_positions


def lbgm_synthesis(train_data, val_data, test_data, train_positions,
                   val_positions, test_positions, names):
    random_state = 42
    train_data = create_target2(total_data=train_data,
                                positions=train_positions)
    pdb.set_trace()
    #  # 对目标y进行排序转换，增强鲁棒性
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data[names])
    y_transformed = rank_transform(train_data['target'])
    pdb.set_trace()
    X_train, X_val, y_train, y_val = train_test_split(X_train_scaled,
                                                      y_transformed,
                                                      test_size=0.2,
                                                      shuffle=False)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
    params = {
        #'objective': 'regression_l1',
        'metric': 'mae',  # Mean Absolute Error
        'n_estimators': 200,  # 可以设置一个较大的值，由早停来控制
        'learning_rate': 0.05,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 1,
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        'num_leaves': 31,
        'min_gain_to_split': 0.0,  # 确保任何非负增益都会分裂
        'min_child_samples': 5,  # 大幅减小叶子节点的最小样本数，允许更深的分裂
        'min_child_weight': 1e-5,  # 同样减小
        'verbose': -1,
        'n_jobs': -1,
        'seed': random_state,
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': 1
    }
    pdb.set_trace()
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(10, verbose=False)]  # 早停是关键  
    )

    feature_importances = pd.Series(model.feature_importance(), index=names)
    print("子策略重要性 (Top 5):\n",
          feature_importances.sort_values(ascending=False).head())

    positions = pd.concat([train_positions, val_positions, test_positions],
                          axis=0)
    positions = positions.set_index('trade_time')
    raw_meta_signal = model.predict(positions,
                                    num_iteration=model.best_iteration)
    meta_positions = rank_normalize_signal(
        pd.Series(raw_meta_signal, index=positions.index))
    meta_positions.name = "lgbm"
    return meta_positions


def calcute_fitness(positions,
                    total_data,
                    strategy_settings,
                    base_dirs,
                    key=None):
    name = positions.name
    positions.name = 'pos'
    positions = positions.reset_index()
    positions['code'] = INSTRUMENTS_CODES[g_instruments]
    positions = positions.set_index(['trade_time', 'code']).unstack()
    pnl_in_window = calculate_ful_ts_ret(
        pos_data=positions,
        total_data=total_data,
        strategy_settings=strategy_settings,
        agg=True  # 确保按天聚合
    )
    dirs = os.path.join(os.path.join(base_dirs, 'returns', key)) if isinstance(
        key, str) else os.path.join(os.path.join(base_dirs, 'returns', key))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    pnl_in_window.reset_index().to_feather(
        os.path.join(dirs, "{0}.feather".format(name)))


if __name__ == '__main__':
    method = 'aicso2'
    g_instruments = 'ims'
    task_id = '200037'
    threshold = 1.1

    base_dirs = os.path.join(
        os.path.join('temp', "{}".format(method),
                     PERFORMANCE_MAPPING[str(task_id)],
                     INSTRUMENTS_CODES[g_instruments]))

    ## 选中的策略
    benchmark = [
        'ultron_1751395662731039', 'ultron_1751492470196206',
        'ultron_1751388038959442', 'ultron_1751431447109266'
    ]
    strategy_pool = {
        'bk': benchmark,
        'tst1': benchmark + ['ultron_1751397805025247','ultron_1751431447109266'],
        'tst2': [
            'ultron_1751401005542132', 'ultron_1751414027498169',
            'ultron_1751386610921461', 'ultron_1751388038959442',
            'ultron_1751397805025247', 'ultron_1751431447109266',
            'ultron_1751375993205158', 'ultron_1751460301087405',
            'ultron_1751492470196206', 'ultron_1751385107413126'
        ]
    }
    key = 'tst2'

    names = strategy_pool[key]

    strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[g_instruments]] * 0.05,
        'slippage': 0,  #SLIPPAGE_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[g_instruments]]
    }
    base_dirs = os.path.join(os.path.join('temp', "{}".format(method),
                                          task_id))
    if not os.path.exists(base_dirs):
        os.makedirs(base_dirs)

    programs = load_fitness(base_dirs=base_dirs)
    ## 加载仓位
    positions_res = load_positions(base_dirs=base_dirs, names=names)

    test_positions = merge_positions(positions_res=positions_res, mode='test')
    val_positions = merge_positions(positions_res=positions_res, mode='val')
    train_positions = merge_positions(positions_res=positions_res,
                                      mode='train')

    ## 基础数据
    val_data = load_data(mode='val')
    train_data = load_data(mode='train')
    test_data = load_data(mode='test')

    total_data = pd.concat([train_data, val_data, test_data],
                           axis=0).sort_values(by=['trade_time'])
    total_data = total_data.copy().reset_index().set_index(
        ['trade_time', 'code']).unstack()

    lasso_positions = lasso_regression_synthesis(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_positions=train_positions,
        val_positions=val_positions,
        test_positions=test_positions,
        names=names)

    rigde_positions = rigde_regression_synthesis(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_positions=train_positions,
        val_positions=val_positions,
        test_positions=test_positions,
        names=names)

    random_positions = random_forest_synthesis(train_data=train_data,
                                               val_data=val_data,
                                               test_data=test_data,
                                               train_positions=train_positions,
                                               val_positions=val_positions,
                                               test_positions=test_positions,
                                               names=names)

    calcute_fitness(positions=lasso_positions,
                    total_data=total_data,
                    strategy_settings=strategy_settings,
                    base_dirs=base_dirs,
                    key=key)

    calcute_fitness(positions=rigde_positions,
                    total_data=total_data,
                    strategy_settings=strategy_settings,
                    base_dirs=base_dirs,
                    key=key)

    calcute_fitness(positions=random_positions,
                    total_data=total_data,
                    strategy_settings=strategy_settings,
                    base_dirs=base_dirs,
                    key=key)
