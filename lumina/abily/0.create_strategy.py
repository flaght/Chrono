import os
from dotenv import load_dotenv
import pandas as pd
import ultron.factor.empyrical as empyrical
from lumina.genetic.signal.method import *
from lumina.genetic.strategy.method import *
from lumina.genetic.process import *
from ultron.factor.genetic.geneticist.operators import *
from kdutils.orders import profit_rate, position_next_order_cy
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_pnl, calculate_ful_ts_ret

load_dotenv()

from kdutils.macro import *
from kdutils.file import fetch_file_data

os.environ['INSTRUMENTS'] = 'ims'
g_instruments = os.environ['INSTRUMENTS']

if __name__ == '__main__':
    method = 'aicso2'
    total_data = fetch_file_data(base_path=base_path,
                                 method=method,
                                 g_instruments=g_instruments,
                                 datasets=['train_data','test_data'])
    pdb.set_trace()
    total_data = total_data#.loc[:4000]
    total_data['trade_time'] = pd.to_datetime(total_data['trade_time'])

    strategy_settings = {
        'capital': 10000000,
        'commission': COST_MAPPING[instruments_codes[g_instruments][0]],
        'slippage': SLIPPAGE_MAPPING[instruments_codes[g_instruments][0]],
        'size': CONT_MULTNUM_MAPPING[instruments_codes[g_instruments][0]]
    }

    total_data1 = total_data.set_index(['trade_time'])
    total_data2 = total_data.set_index(['trade_time', 'code']).unstack()

    total_dt = total_data2.copy()

    formual1 = "MRes(10,MPERCENT(20,'tc006_5_10_1'),'oi030_5_10_1')"
    signal_method = 'quantile_signal'
    strategy_method = 'trailing_atr_strategy'
    signal_params = {"threshold": 0.5, "roll_num": 35}
    strategy_params = {"atr_period": 20, "atr_multiplier": 5, "max_volume": 1}
    factors_data = calc_factor(expression=formual1,
                               total_data=total_data1,
                               key='code',
                               indexs=[])

    factors_data1 = factors_data.reset_index().set_index(
        ['trade_time', 'code'])
    pos_data = eval(signal_method)(factor_data=factors_data1, **signal_params)
    pos_data1 = eval(strategy_method)(signal=pos_data,
                                      total_data=total_dt,
                                      **strategy_params)

    total_data2['trade_vol', total_data2['open'].columns[0]] = (
        strategy_settings['capital'] / total_data2['open'] /
        strategy_settings['size'])

    pdb.set_trace()
    orders = position_next_order_cy(pos_data=pos_data1,
                                    market_data=total_data2,
                                    commission=strategy_settings['commission'],
                                    slippage=strategy_settings['slippage'])
    #fitness = win_rate(orders=orders)

    df = calculate_ful_ts_ret(pos_data=pos_data1,
                              total_data=total_data2,
                              strategy_settings=strategy_settings)
    r1 = profit_rate(orders=orders)
    r2 =  df.sum().values[0]
    assert(round(r1, 4) == round(r2, 4))
