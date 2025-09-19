import numpy as np
from kdutils.macro2 import *
from kdutils.common import *
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret


def calcute_fitness(positions,
                    total_data,
                    strategy_settings,
                    instruments,
                    base_dirs,
                    key=None):
    save_positions = positions.copy()
    name = positions.name
    positions.name = 'pos'
    positions = positions.reset_index()
    positions['code'] = INSTRUMENTS_CODES[instruments]
    positions = positions.set_index(['trade_time', 'code']).unstack()

    pnl_in_window = calculate_ful_ts_ret(
        pos_data=positions,
        total_data=total_data,
        strategy_settings=strategy_settings,
        agg=True  # 确保按天聚合
    )

    ### 存储绩效
    dirs = os.path.join(os.path.join(base_dirs, 'returns', key)) if isinstance(
        key, str) else os.path.join(os.path.join(base_dirs, 'returns', key))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    print(dirs)
    pnl_in_window.reset_index().to_feather(
        os.path.join(dirs, "{0}.feather".format(name)))

    ### 存储仓位
    dirs = os.path.join(os.path.join(
        base_dirs, 'positions', key)) if isinstance(
            key, str) else os.path.join(
                os.path.join(base_dirs, 'positions', key))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    print(dirs)
    save_positions.reset_index().to_feather(
        os.path.join(dirs, "{0}.feather".format(name)))


def split_positions(base_dirs, positions, time_periods, key):
    name = positions.name
    train_positions = positions.loc[
        time_periods['train_time'][0]:time_periods['train_time'][1]]
    val_positions = positions.loc[
        time_periods['val_time'][0]:time_periods['val_time'][1]]

    test_positions = positions.loc[
        time_periods['test_time'][0]:time_periods['test_time'][1]]

    dirs = os.path.join(os.path.join(
        base_dirs, 'positions', key)) if isinstance(
            key, str) else os.path.join(
                os.path.join(base_dirs, 'positions', key))
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    train_positions.reset_index().to_feather(
        os.path.join(dirs, "{0}_train.feather".format(name)))

    val_positions.reset_index().to_feather(
        os.path.join(dirs, "{0}_val.feather".format(name)))

    test_positions.reset_index().to_feather(
        os.path.join(dirs, "{0}_test.feather".format(name)))


def fetch_all_data(method, instruments, task_id):

    val_data = load_data(method=method,
                         instruments=instruments,
                         task_id=task_id,
                         mode='val')
    train_data = load_data(method=method,
                           instruments=instruments,
                           task_id=task_id,
                           mode='train')
    test_data = load_data(method=method,
                          instruments=instruments,
                          task_id=task_id,
                          mode='test')
    pdb.set_trace()
    val_data.index = pd.to_datetime(val_data.index)
    train_data.index = pd.to_datetime(train_data.index)
    test_data.index = pd.to_datetime(test_data.index)
    #train_data = train_data.loc[train_data.index[0:2000]]
    #test_data = test_data.loc[test_data.index[0:1000]]
    #val_data = val_data.loc[val_data.index[0:1000]]

    total_data = pd.concat([train_data, val_data, test_data],
                           axis=0).sort_values(by=['trade_time'])
    total_data = total_data.copy().reset_index().set_index(
        ['trade_time', 'code']).unstack()
    return total_data, train_data, val_data, test_data


def process_positions(positions_res, key):
    test_positions = merge_positions(positions_res=positions_res, mode='test')
    val_positions = merge_positions(positions_res=positions_res, mode='val')
    train_positions = merge_positions(positions_res=positions_res,
                                      mode='train')

    pdb.set_trace()
    if any(df.empty
           for df in [train_positions, val_positions, test_positions]):
        print(f"警告: 策略池 '{key}' 的训练/验证/测试仓位数据不完整，跳过此池。")
        return None
    
    #test_positions = test_positions.loc[0:1000]
    #val_positions = val_positions.loc[0:1000]
    #train_positions = train_positions.loc[0:2000]
    positions = pd.concat([train_positions, val_positions, test_positions],
                          axis=0).sort_values(by=['trade_time'])
    positions = positions.set_index('trade_time')
    return positions, train_positions, val_positions, test_positions


def create_target1(total_data,
                   positions,
                   instruments,
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



def create_target2(total_data, positions, instruments, price_col='close'):
    data = total_data.sort_values(
        by=['code', 'trade_time']).copy().reset_index()
    data = data[['trade_time', 'code',
                 price_col]].set_index(['trade_time', 'code'])[price_col]
    data = data.unstack()
    future_log_return = np.log((data / data.shift(1))).shift(-1)
    y_target = future_log_return.dropna()  #np.sign(future_log_return)
    y_target = y_target.reset_index().rename(
        columns={INSTRUMENTS_CODES[instruments]: 'target'})
    return positions.merge(y_target, on=['trade_time'])


def rank_transform(series: pd.Series) -> pd.Series:
    ranked = series.rank(pct=True)
    return (ranked * 2) - 1


def rank_normalize_signal(raw_signal: pd.Series) -> pd.Series:
    """通过排序并映射到[-1, 1]区间来进行标准化。"""
    ranked_signal = raw_signal.rank(pct=True)
    normalized_signal = (ranked_signal * 2) - 1
    return normalized_signal
