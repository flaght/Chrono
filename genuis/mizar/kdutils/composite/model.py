from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

from kdutils.composite.prepare import *


def composite1(train_data, val_data, test_data, train_positions, val_positions,
               test_positions, instruments, names, params, model_class):
    pdb.set_trace()
    train_data = create_target2(total_data=train_data,
                                positions=train_positions,
                                instruments=instruments)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data[names])
    y_train = train_data['target']

    model = model_class(**params)
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
    return meta_positions


def lasso_regression1(train_data,
                      val_data,
                      test_data,
                      train_positions,
                      val_positions,
                      test_positions,
                      instruments,
                      names,
                      params,
                      key=None):
    meta_positions = composite1(train_data=train_data,
                                val_data=val_data,
                                test_data=test_data,
                                train_positions=train_positions,
                                val_positions=val_positions,
                                test_positions=test_positions,
                                instruments=instruments,
                                names=names,
                                model_class=LassoCV,
                                params=params)
    meta_positions.name = 'lasso' if not isinstance(
        key, str) else "lasso_{}".format(key)
    return meta_positions


def rigde_regression1(train_data,
                      val_data,
                      test_data,
                      train_positions,
                      val_positions,
                      test_positions,
                      instruments,
                      names,
                      params,
                      key=None):
    meta_positions = composite1(train_data=train_data,
                                val_data=val_data,
                                test_data=test_data,
                                train_positions=train_positions,
                                val_positions=val_positions,
                                test_positions=test_positions,
                                names=names,
                                instruments=instruments,
                                model_class=RidgeCV,
                                params=params)
    meta_positions.name = 'lasso' if not isinstance(
        key, str) else "lasso_{}".format(key)
    return meta_positions
