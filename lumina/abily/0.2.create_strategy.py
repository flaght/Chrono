import os, time
from dotenv import load_dotenv
import pandas as pd
import ultron.factor.empyrical as empyrical
from lumina.genetic.signal.method import *
from lumina.genetic.strategy.method import *
from lumina.genetic.process import *
from ultron.factor.genetic.geneticist.operators import *
#from kdutils.orders import profit_rate, profit_std, position_next_order_cy, position_next_order_nb, position_next_order_oi
from lumina.genetic.fusion.orders import profit_std, profit_rate, position_next_order_cy, position_next_order_nb, position_next_order_oi
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_pnl, calculate_ful_ts_ret

load_dotenv()

from kdutils.macro2 import *
from kdutils.file import fetch_file_data

if __name__ == '__main__':
    method = 'aicso2'
    g_instruments = 'ims'
    total_data = fetch_file_data(
        base_path=base_path,
        method=method,
        g_instruments=g_instruments,
        datasets=['train_data', 'val_data', 'test_data'])
    pdb.set_trace()
    total_data = total_data  #.loc[:4000]
    total_data['trade_time'] = pd.to_datetime(total_data['trade_time'])

    strategy_settings = {
        'capital': 10000000,
        'commission': COST_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'slippage': SLIPPAGE_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[g_instruments]]
    }

    total_data1 = total_data.set_index(['trade_time'])
    total_data2 = total_data.set_index(['trade_time', 'code']).unstack()

    total_dt = total_data2.copy()

    formual1 = "MMIN(14,MIChimoku(6,WMA(10,MIChimoku(6,MIR(20,MPRO(8,'tn001_10_15_1')),MSmart(14,FLOOR('tc007_10_15_1'),MDPO(6,'tc014_5_5_10_0')))),MPRO(8,'tn009_5_10_0_1')))"
    signal_method = 'adaptive_signal'
    strategy_method = 'volume_weighted_strategy'
    signal_params = {'threshold': 0.6, 'roll_num': 60}
    strategy_params = {'roll_num': 30, 'max_volume': 1}
    factors_data = calc_factor(expression=formual1,
                               total_data=total_data1,
                               key='code',
                               indexs=[])
    pdb.set_trace()
    key_name = 'open'
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
    strategy_settings['commission'] = 0.00023
    orders = position_next_order_cy(pos_data=pos_data1,
                                    market_data=total_data2,
                                    commission=strategy_settings['commission'],
                                    slippage=strategy_settings['slippage'],
                                    name=key_name)
    pdb.set_trace()
    profit_rate_val = profit_rate(orders=orders)  #profit_std(orders=orders)
    #profit_std(orders=orders, n_sigma=3)
    #fitness = win_rate(orders=orders)

    df = calculate_ful_ts_ret(pos_data=pos_data1,
                              total_data=total_data2,
                              strategy_settings=strategy_settings,
                              name=key_name,
                              agg=False)
    df1 = calculate_ful_ts_pnl(pos_data=pos_data1,
                               total_data=total_data2,
                               strategy_settings=strategy_settings)
    print(f"Profit Rate: {profit_rate_val}")
    #r1 = profit_rate(orders=orders)
    #r2 = df.sum().values[0]
    #assert (round(r1, 4) == round(r2, 4))
