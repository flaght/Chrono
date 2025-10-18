import itertools
from lumina.genetic.signal.method import create_adaptive_muster
from lumina.genetic.strategy.method import create_meanrevert_muster
from lumina.genetic.signal.method import *
from lumina.genetic.strategy.method import *
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret
from lumina.genetic.process import *
from ultron.factor.genetic.geneticist.operators import *
import ultron.factor.empyrical as empyrical

from dotenv import load_dotenv

load_dotenv()

from kdutils.common import *
from kdutils.macro2 import *


def compare_fitness_rate(column, method, instruments, times_info, task_id,
                         base_dirs):
    strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]],
        'slippage': 0,
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }

    print(column)

    def calcute_fitness(column, name, method, instruments, task_id):
        backup_cycle = 1
        total_data = fetch_temp_data(
            method=method,
            task_id=task_id,
            instruments=instruments,
            datasets=name if isinstance(name, list) else [name])

        total_data = total_data.drop_duplicates(
            subset=['trade_time', 'code']).reset_index(drop=True)
        ## 计算
        total_data1 = total_data.set_index(['trade_time'])
        total_data2 = total_data.set_index(['trade_time', 'code']).unstack()

        expression = column['formual']
        signal_method = column['signal_method']
        strategy_method = column['strategy_method']
        signal_params = column['signal_params']
        signal_params = {
            key: value
            for key, value in signal_params.items() if value is not None
        }
        strategy_params = column['strategy_params']
        strategy_params = {
            key: value
            for key, value in strategy_params.items() if value is not None
        }

        ### 保持和挖掘一致， 要做inf值处理，要做极小值处理
        factor_data = calc_factor(expression=expression,
                                  total_data=total_data1,
                                  indexs=[],
                                  key='code')
        factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
        factor_data['transformed'] = np.where(
            np.abs(factor_data.transformed.values) > 0.000001,
            factor_data.transformed.values, np.nan)
        factor_data = factor_data.loc[factor_data.index.unique()
                                      [backup_cycle:]]

        factors_data1 = factor_data.reset_index().set_index(
            ['trade_time', 'code'])

        cycle_total_data = total_data.copy()
        cycle_total_data = cycle_total_data.loc[
            cycle_total_data.index.unique()[backup_cycle:]]

        total_data1 = cycle_total_data.reset_index().set_index(
            ['trade_time', 'code']).unstack()

        pos_data = eval(signal_method)(factor_data=factors_data1,
                                       **signal_params)
        pos_data1 = eval(strategy_method)(signal=pos_data,
                                          total_data=total_data1,
                                          **strategy_params)

        df = calculate_ful_ts_ret(pos_data=pos_data1,
                                  total_data=total_data2,
                                  strategy_settings=strategy_settings)
        returns = df['a_ret']
        fitness = empyrical.sharpe_ratio(returns=returns,
                                         period=empyrical.DAILY)
        return pos_data1, fitness, df

    ## 主要用于校验和挖掘过程fitness是否一致
    #fitness, df = calcute_fitness(strategy=strategy,
    #                              name=['train'],
    #                              method=method,
    #                              instruments=instruments)
    posisition, fitness, df = calcute_fitness(column=column,
                                              name=['train', 'val', 'test'],
                                              method=method,
                                              instruments=instruments,
                                              task_id=task_id)
    train_df = df[(df.index >= times_info['train_time'][0])
                  & (df.index <= times_info['train_time'][1])]
    train_fitness = empyrical.sharpe_ratio(returns=train_df['a_ret'],
                                           period=empyrical.DAILY)

    val_df = df[(df.index >= times_info['val_time'][0])
                & (df.index <= times_info['val_time'][1])]
    val_fitness = empyrical.sharpe_ratio(returns=val_df['a_ret'],
                                         period=empyrical.DAILY)

    test_df = df[(df.index >= times_info['test_time'][0])
                 & (df.index <= times_info['test_time'][1])]
    test_fitness = empyrical.sharpe_ratio(returns=test_df['a_ret'],
                                          period=empyrical.DAILY)

    train_fitness_abs = abs(train_fitness)
    if train_fitness_abs < 1e-8:  # 避免除以零
        val_retention = 0
        test_retention = 0
    else:
        val_retention = val_fitness / train_fitness_abs
        test_retention = test_fitness / train_fitness_abs
    #val_rate = math.fabs(val_fitness -
    #                     train_fitness) / math.fabs(train_fitness)
    #test_rate = math.fabs(test_fitness -
    #                      train_fitness) / math.fabs(train_fitness)

    posisition.columns = ['pos']
    train_posisition = posisition[
        (posisition.index >= times_info['train_time'][0])
        & (posisition.index <= times_info['train_time'][1])]

    val_posisition = posisition[
        (posisition.index >= times_info['val_time'][0])
        & (posisition.index <= times_info['val_time'][1])]

    test_posisition = posisition[
        (posisition.index >= times_info['test_time'][0])
        & (posisition.index <= times_info['test_time'][1])]

    ## 保存收益
    dirs = os.path.join(os.path.join(base_dirs, 'returns'))
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    df.reset_index().to_feather(
        os.path.join(dirs, "{0}.feather".format(column['name'])))

    ## 保存仓位
    dirs = os.path.join(os.path.join(base_dirs, 'positions'))
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    train_posisition.reset_index().to_feather(
        os.path.join(dirs, "{0}_train.feather".format(column['name'])))

    val_posisition.reset_index().to_feather(
        os.path.join(dirs, "{0}_val.feather".format(column['name'])))

    test_posisition.reset_index().to_feather(
        os.path.join(dirs, "{0}_test.feather".format(column['name'])))
    #pdb.set_trace()
    return {
        'name': column['name'],
        'all_fitness': fitness,
        'train_fitness': train_fitness,
        'val_fitness': val_fitness,
        'test_fitness': test_fitness,
        'val_retention': val_retention,
        'test_retention': test_retention
    }


### 批量生成策略
@add_process_env_sig
def run_position(target_column, method, instruments, times_info, task_id,
                 base_dirs):
    position_data = run_process(target_column=target_column,
                                callback=compare_fitness_rate,
                                method=method,
                                instruments=instruments,
                                times_info=times_info,
                                task_id=task_id,
                                base_dirs=base_dirs)
    return position_data


if __name__ == '__main__':
    method = 'aicso0'
    instruments = 'ims'
    task_id = '200036'
    threshold = 1.1

    times_info = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)

    strategy_dt = fetch_strategy1(task_id=task_id,
                                  method=method,
                                  instruments=instruments,
                                  threshold=threshold)
    strategy1 = strategy_dt.loc[0]  #['formual']

    strategy_sets = create_meanrevert_muster()
    signal_sets = create_adaptive_muster()
    res = []
    k_split = 8
    base_dirs = os.path.join(
        os.path.join('temp', "{}".format(method), task_id, 'experiment'))
    if not os.path.exists(base_dirs):
        os.makedirs(base_dirs)

    strategies_infos = []
    strategies_list = list(itertools.product(signal_sets, strategy_sets))
    for i, strategy in enumerate(strategies_list):
        s1 = {
            'name': 'ultron_{0}_1752177076449476'.format(i),
            'formual': "MA(2,'rv002_10_15_1_2')",
            'strategy_method': strategy[1].name,
            'strategy_params': strategy[1].params,
            'signal_method': strategy[0].name,
            'signal_params': strategy[0].params
        }
        strategies_infos.append(s1)
    strategies_infos = strategies_infos
    process_list = split_k(k_split, strategies_infos)
    res = create_parellel(process_list=process_list,
                          callback=run_position,
                          method=method,
                          instruments=instruments,
                          times_info=times_info,
                          task_id=task_id,
                          base_dirs=base_dirs)
    pdb.set_trace()
    res = list(itertools.chain.from_iterable(res))
    pdb.set_trace()
    pd.DataFrame(res).to_feather(os.path.join(base_dirs, "fitness.feather"))
