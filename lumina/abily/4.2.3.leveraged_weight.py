### 主要针对杠杆为1的情况合成
import pandas as pd
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv

load_dotenv()
from kdutils.macro2 import *
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret


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
    pdb.set_trace()
    fitness_file = os.path.join(base_dirs, "fitness.feather")
    fitness_pd = pd.read_feather(fitness_file)

    return fitness_pd


def load_data(mode='train'):
    pdb.set_trace()
    filename = os.path.join(base_path, method, instruments, 'level2',
                            '{0}_data.feather'.format(mode))
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    return factors_data.set_index('trade_time')


def create_target2(total_data,
                   positions,
                   instruments,
                   price_col='close',
                   neutral_threshold=0.00023 * 0.5):
    data = total_data.sort_values(
        by=['code', 'trade_time']).copy().reset_index()
    data = data[['trade_time', 'code',
                 price_col]].set_index(['trade_time', 'code'])[price_col]
    data = data.unstack()
    future_log_return = np.log((data / data.shift(1))).shift(-1)
    y_target = future_log_return.dropna()
    y_target = y_target.reset_index().rename(
        columns={INSTRUMENTS_CODES[instruments]: 'target'})

    y_target = y_target.set_index('trade_time')['target']
    y = pd.Series(0, index=y_target.index, name='target')
    y[y_target > neutral_threshold] = 1
    y[y_target < -neutral_threshold] = -1
    y_target = y
    return positions.merge(y_target.reset_index(), on=['trade_time'])


def lbgm_synthesis(train_data, val_data, test_data, train_positions,
                   val_positions, test_positions, names, instruments):

    random_state = 42
    ## 训练集
    train_data = create_target2(total_data=train_data,
                                positions=train_positions,
                                instruments=instruments)
    train_data = train_data.set_index('trade_time')
    y_ternary = train_data['target']
    y_ternary = y_ternary.map({-1: 0, 0: 1, 1: 2})
    X_train = train_data[names]
    lgb_train = lgb.Dataset(X_train.values.astype(np.float32),
                            label=y_ternary.values.astype(np.float32))
    
    ## 校验集
    val_data = create_target2(total_data=val_data,
                              positions=val_positions,
                              instruments=instruments)
    val_data = val_data.set_index('trade_time')
    y_ternary = val_data['target']
    y_ternary = y_ternary.map({-1: 0, 0: 1, 1: 2})
    X_val = val_data[names]
    lgb_val = lgb.Dataset(X_val.values.astype(np.float32),
                          label=y_ternary.values.astype(np.float32))
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'n_estimators': 500,  # 可以设置一个合理的最大值
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'num_leaves': 15,  # 对于弱信号，简单的树可能更好
        'min_child_samples': 50,  # 提高叶子节点的最小样本数，防止过拟合
        'min_gain_to_split': 0.0,  # 允许任何正增益分裂
        'n_jobs': -1,
        'seed': 42,
        'device': 'gpu',
        'verbose': 1
    }

    print("X_train std:{} and std num:{}".format(X_train.values.std(),
                                                 X_train.std().sum()))
    print("y:{0}".format(y_ternary.nunique()))

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(10, verbose=True, first_metric_only=True)
        ]  # 早停是关键  
    )
    feature_importances = pd.Series(model.feature_importance(), index=names)
    print("子策略重要性 (Top 5):\n",
          feature_importances.sort_values(ascending=False).head())

    print("正在生成最终的离散目标仓位...")

    ## 测试集
    test_data = create_target2(total_data=test_data,
                               positions=test_positions,
                               instruments=instruments)
    test_data = test_data.set_index('trade_time')
    y_ternary = test_data['target']
    y_ternary = y_ternary.map({-1: 0, 0: 1, 1: 2})
    X_test = test_data[names]
    lgb_test = lgb.Dataset(X_test, label=y_ternary)

    # 使用 .predict() 会直接输出每个样本概率最高的那个类别 {0, 1, 2}
    predicted_labels = model.predict(X_test,
                                     num_iteration=model.best_iteration)

    position_map = {0: -1, 1: 0, 2: 1}
    predicted_probs = model.predict(X_test, num_iteration=model.best_iteration)

    final_predicted_labels = np.argmax(predicted_probs, axis=1)

    final_positions = pd.Series(final_predicted_labels,
                                index=X_test.index).map(position_map)
    final_positions.name = "meta_position_lgbm_multiclass"
    return final_positions


if __name__ == '__main__':
    method = 'aicso2'
    instruments = 'ims'
    task_id = '200037'
    threshold = 1.1

    base_dirs = os.path.join(
        os.path.join('temp', "{}".format(method), str(task_id)))

    strategy_pool = {
        'all': [
            'ultron_1751385041297455', 'ultron_1751375993205158',
            'ultron_1751401005542132', 'ultron_1751375798567305',
            'ultron_1751397805025247', 'ultron_1751391914745352',
            'ultron_1751431447109266', 'ultron_1751388038959442',
            'ultron_1751389839279277'
        ]
    }
    key = 'all'

    names = strategy_pool[key]

    strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]] * 0.05,
        'slippage': 0,  #SLIPPAGE_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }
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
    pdb.set_trace()
    lbgm_synthesis(train_data=train_data,
                   val_data=val_data,
                   test_data=test_data,
                   train_positions=train_positions,
                   val_positions=val_positions,
                   test_positions=test_positions,
                   names=names,
                   instruments=instruments)
