import os, pdb, optuna, empyrical
import pandas as pd
import numpy as np
import lightgbm as lgb
from optuna.integration import LightGBMPruningCallback
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret


def load_data(mode='train'):
    base_path = './records'
    filename = os.path.join(base_path, method, g_instruments, 'level2',
                            '{0}_data.feather'.format(mode))
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    return factors_data.set_index('trade_time')


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
        #print(name)
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


def fetch_data(base_dirs, names=[]):
    programs = load_fitness(base_dirs=base_dirs)
    features = programs['name'].tolist()
    ## 加载仓位
    positions_res = load_positions(base_dirs=base_dirs, names=features)
    test_positions = merge_positions(positions_res=positions_res, mode='test')
    val_positions = merge_positions(positions_res=positions_res, mode='val')
    train_positions = merge_positions(positions_res=positions_res,
                                      mode='train')
    positions = pd.concat([train_positions, val_positions, test_positions],
                          axis=0).sort_values(by=['trade_time'])
    positions = positions.set_index('trade_time')

    ## 加载数据
    val_data = load_data(mode='val')
    train_data = load_data(mode='train')
    test_data = load_data(mode='test')
    total_data = pd.concat([train_data, val_data, test_data],
                           axis=0).sort_values(by=['trade_time'])

    total_data = total_data.copy().reset_index().set_index(
        ['trade_time', 'code']).unstack()

    return train_positions,val_positions,test_positions,\
            positions,train_data,val_data,test_data,\
            total_data,features


def create_target2(total_data,
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
    y_target = y_target.map({-1: 0, 0: 1, 1: 2})
    return positions.merge(y_target, on=['trade_time'])


def train_lgbm_for_optuna(params, X_train, y_train, X_val, y_val,
                          total_data_val, pruning_callback, strategy_settings):
    """
    一个接收参数字典并执行一次LGBM训练的函数。
    它返回在验证集上的性能指标。
    """
    params.pop('n_estimators', None)

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    pdb.set_trace()
    try:
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,  # 给一个足够大的轮数
            valid_sets=[lgb_val],
            valid_names=['val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                pruning_callback  # 加入剪枝回调
            ])
    except Exception as e:
        # 如果某些参数组合导致错误，返回一个极差的分数
        print(f"训练失败，参数: {params}, 错误: {e}")
        return -np.inf

    # --- 3. 在验证集上生成目标仓位 ---
    predicted_probs = model.predict(X_val, num_iteration=model.best_iteration)
    predicted_labels = np.argmax(predicted_probs, axis=1)
    position_map = {0: -1, 1: 0, 2: 1}
    val_positions = pd.Series(predicted_labels,
                              index=X_val.index).map(position_map)

    pos_data_val = val_positions.to_frame('pos').unstack()

    pnl_df_val = calculate_ful_ts_ret(
        pos_data=pos_data_val,
        total_data=total_data_val.unstack(),  # 假设回测函数需要unstacked
        strategy_settings=custom_params['strategy_settings'])

    # 使用验证集夏普比率作为最终的优化目标
    sharpe_ratio_val = empyrical.sharpe_ratio(pnl_df_val['a_ret'],
                                              period='daily')

    # 确保返回一个有限的浮点数
    return sharpe_ratio_val if not np.isnan(sharpe_ratio_val) else -np.inf


class SharpePruningCallback:

    def __init__(self, trial: optuna.Trial, X_val, total_data_val, val_data,
                 strategy_settings):
        self.trial = trial
        self.X_val = X_val
        self.total_data_val = total_data_val
        self.val_data = val_data
        self.strategy_settings = strategy_settings
        self.best_sharpe = -np.inf

    def __call__(self, env: lgb.callback.CallbackEnv):
        # 在每一轮迭代结束时，这个函数会被调用
        model = env.model
        current_iteration = env.iteration

        # 1. 在验证集上生成当前模型的仓位
        predicted_probs = model.predict(self.X_val,
                                        num_iteration=current_iteration + 1)
        predicted_labels = np.argmax(predicted_probs, axis=1)
        position_map = {0: -1, 1: 0, 2: 1}

        val_positions = pd.Series(
            predicted_labels,
            index=self.val_data['trade_time']).map(position_map)

        val_positions.name = 'pos'
        val_positions = val_positions.reset_index()
        val_positions['code'] = 'IM'
        val_positions = val_positions.set_index(['trade_time',
                                                 'code']).unstack()
        # 2. 回测并计算夏普比率
        pnl_df = calculate_ful_ts_ret(pos_data=val_positions,
                                      total_data=self.total_data_val,
                                      strategy_settings=self.strategy_settings)
        current_sharpe = empyrical.sharpe_ratio(pnl_df['a_ret'],
                                                period='daily')
        current_sharpe = current_sharpe if not np.isnan(
            current_sharpe) else -np.inf

        # 3. 将中间结果报告给trial，用于剪枝
        self.trial.report(current_sharpe, step=current_iteration)

        # 4. 检查是否应该剪枝
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()


def objective_final(trial: optuna.Trial, X_train, y_train, X_val, y_val,
                    total_data_val, val_data, strategy_settings):
    """
    一个完整的、健壮的Optuna目标函数。
    """
    # 1. 定义超参数搜索空间
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbose': -1,
        'n_jobs': -1,
        'device': 'gpu',
        'learning_rate': trial.suggest_float('learning_rate',
                                             0.01,
                                             0.2,
                                             log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
    }

    # 2. 准备数据和回调
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    # 核心修正：明确指定 valid_name
    #pruning_callback = LightGBMPruningCallback(trial, "multi_logloss", valid_name="val")
    pruning_callback = SharpePruningCallback(trial, X_val, total_data_val,
                                             val_data, strategy_settings)

    # 3. 训练模型
    try:
        model = lgb.train(params,
                          lgb_train,
                          num_boost_round=1000,
                          valid_sets=[lgb_train, lgb_val],
                          valid_names=['train', 'val'],
                          callbacks=[
                              lgb.early_stopping(30, verbose=False),
                              pruning_callback
                          ])
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"训练失败: {e}")
        return -np.inf

    # 4. 在验证集上预测并回测
    predicted_probs = model.predict(X_val, num_iteration=model.best_iteration)
    predicted_labels = np.argmax(predicted_probs, axis=1)
    position_map = {0: -1, 1: 0, 2: 1}
    val_positions = pd.Series(predicted_labels,
                              index=val_data['trade_time']).map(position_map)

    val_positions.name = 'pos'
    val_positions = val_positions.reset_index()
    val_positions['code'] = 'IM'
    val_positions = val_positions.set_index(['trade_time', 'code']).unstack()

    # 假设 total_data_val 的索引是 DatetimeIndex, 列是 MultiIndex
    pnl_df_val = calculate_ful_ts_ret(pos_data=val_positions,
                                      total_data=total_data_val,
                                      strategy_settings=strategy_settings)

    sharpe_ratio_val = empyrical.sharpe_ratio(pnl_df_val['a_ret'],
                                              period='daily')
    return sharpe_ratio_val if not np.isnan(sharpe_ratio_val) else -np.inf


def run(base_dirs, names=[]):
    train_positions,val_positions,test_positions,\
        positions,train_data,val_data,test_data,\
        total_data,features = fetch_data(base_dirs=base_dirs, names=names)

    pdb.set_trace()
    total_data_val = val_data.reset_index().copy().set_index(
        ['trade_time', 'code']).unstack()

    strategy_settings = {
        'commission': 0. * 0.05,
        'slippage': 0,  #SLIPPAGE_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'size': 200
    }

    train_data = create_target2(total_data=train_data,
                                positions=train_positions)
    val_data = create_target2(total_data=val_data, positions=val_positions)

    train_matrix = train_data[['target'] + features].values
    val_matrix = val_data[['target'] + features].values
    pdb.set_trace()
    y_train = train_matrix[:, 0]
    X_train = train_matrix[:, 1:]

    y_val = val_matrix[:, 0]
    X_val = val_matrix[:, 1:]

    study = optuna.create_study(direction='maximize',
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_final(
        trial, X_train, y_train, X_val, y_val, total_data_val, val_data,
        strategy_settings),
                   n_trials=50)


if __name__ == '__main__':
    method = 'aicso2'
    g_instruments = 'ims'
    task_id = '200037'
    threshold = 1.1

    names = []
    base_dirs = os.path.join(os.path.join('temp', "{}".format(method),
                                          task_id))

    run(base_dirs=base_dirs, names=names)
