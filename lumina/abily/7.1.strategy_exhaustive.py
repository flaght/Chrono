### 不带寻优，穷举方式 遍历各个指定参数 计算绩效

import pdb, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret
from lumina.genetic.geneticist.mutation import Generator
from lumina.genetic.geneticist.mutation import Actuator
from lumina.genetic.signal import signal_mapping
from lumina.genetic.strategy import strategy_mapping
import ultron.factor.empyrical as empyrical
from kdutils.macro2 import *


def callback_fitness(factor_data, pos_data, total_data, signal_method,
                     strategy_method, factor_sets, custom_params,
                     default_value):
    strategy_settings = custom_params['strategy_settings']
    df = calculate_ful_ts_ret(pos_data=pos_data,
                              total_data=total_data,
                              strategy_settings=strategy_settings)
    ### 值有异常 绝对值大于1
    returns = df['a_ret']
    #empyrical.cagr(returns=returns, period=empyrical.DAILY)
    fitness = empyrical.sharpe_ratio(returns=returns, period=empyrical.DAILY)
    return fitness


strategy_info = {
    'formual': "MIChimoku(2,RSI(16,'dv001_5_10_1'),MDPO(2,'cj010_5_10_0'))",
    'strategy_method': 'trailing_atr_strategy',
    'strategy_params': {
        'max_volume': 1,
        'atr_multiplier': 7.0,
        'atr_period': 10,
    },
    'signal_method': 'mean_signal',
    'signal_params': {
        'roll_num': 60,
        'threshold': 0.4
    }
}
# a. 时间周期
window_sets1 = [2, 4, 8]
window_sets2 = [14, 16, 18]
window_sets3 = [2, 4, 8]

# b. 创建带参数的信号函数列表
rolling_sets = [50, 60, 70]
threshold_sets = [0.2, 0.4, 0.6]
signal_functions = signal_mapping[strategy_info['signal_method']](
    rolling_sets=rolling_sets, threshold_sets=threshold_sets)

# c. 创建带参数的策略函数列表
atr_period_sets = [8, 10, 12]
atr_multiplier_sets = [6.5, 7, 7.5]
maN_sets = [40, 60, 80]
pdb.set_trace()
strategy_functions = strategy_mapping[strategy_info['strategy_method']](
    atr_multiplier_sets=atr_multiplier_sets,
    atr_period_sets=atr_period_sets,
    maN_sets=maN_sets)

generator = Generator(signal_functions=signal_functions,
                      strategy_functions=strategy_functions)

### 模式一
tuned_recipes = generator.tune_parameters(
    base_info=strategy_info,
    factor_params_space={
        'RSI': {
            'param_0': window_sets2
        },  # 穷举RSI的第一个参数 (周期)
        'MDPO': {
            'param_0': window_sets3
        },  # 穷举MDPO的第一个参数 (周期)
        'MIChimoku': {
            'param_0': window_sets1
        }
    })
#tuned_recipes = tuned_recipes[0:40]
### 模式二
'''
explored_recipes = generator.explore_structures(base_info=strategy_info,
                                                period_params=windows,
                                                factor_max_depth=3)
'''

instruments = 'ims'
method = 'aicso0'
strategy_settings = {
    #'capital': 10000000,
    'mode': COST_MODE_MAPPING[INSTRUMENTS_CODES[instruments]],
    'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]],
    'slippage': SLIPPAGE_MAPPING[INSTRUMENTS_CODES[instruments]],
    'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
}

filename = os.path.join(
    base_path, method, instruments,
    DATAKIND_MAPPING[str(INDEX_MAPPING[INSTRUMENTS_CODES[instruments]])],
    'train_data.feather')

factors_data = pd.read_feather(filename).sort_values(by=['trade_time', 'code'])
factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
factors_data = factors_data.set_index('trade_time')
factor_columns = [
    col for col in factors_data.columns if col not in [
        'trade_time', 'code', 'close', 'high', 'low', 'open', 'value',
        'volume', 'openint', 'vwap'
    ]
]

strategy_settings = {
    #'capital': 10000000,
    'mode': COST_MODE_MAPPING[INSTRUMENTS_CODES[instruments]],
    'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]],
    'slippage': SLIPPAGE_MAPPING[INSTRUMENTS_CODES[instruments]],
    'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
}
rootid = 200036
configure = {
    'rootid': rootid,
    'backup_cycle': 1,
    'coverage_rate': 0.7,
    'custom_params': {
        'g_instruments': instruments,
        'dethod': method,
        'strategy_settings': strategy_settings,
        'task_id': rootid,
        'method': PERFORMANCE_MAPPING[str(rootid)],
    }
}

actuator = Actuator(k_split=32, callback_fitness=callback_fitness)
population = actuator.calculate(total_data=factors_data,
                                strategies_infos=tuned_recipes,
                                factor_columns=factor_columns,
                                configure=configure)
results = [p.output() for p in population]
results = pd.DataFrame(results)
results = results[[
    'name', 'method', 'formual', 'raw_fitness', 'final_fitness',
    'strategy_method', 'strategy_params', 'signal_method', 'signal_params',
    'alpha', 'penalty', 'max_corr'
]]
