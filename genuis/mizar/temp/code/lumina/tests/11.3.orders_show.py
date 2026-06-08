import sys, os, datetime, pdb

sys.path.insert(0, os.path.abspath('../'))
import pandas as pd

from lumina.empyrical.orders.adapter import plot_orders

#orders_file = '/workspace/worker/pj/Chrono/records/phecda/files/hedge041_trader/trader/eval/3/trader_values_eval_trader.csv'
#market_file = '/workspace/data/dev/kd/evolution/nn/phecda/aicso3/normal/ims/rolling/normal_factors3/o2o_1/5_60_5_1_0/normal_factors_test_47.feather'
orders_file = './trader_memory.csv'
market_file = './kl_pd.feather'
pdb.set_trace()
orders_data = pd.read_csv(orders_file, index_col=0)
market_data = pd.read_feather(market_file)

direction = 1
begin_time = '2025-01-16 21:00:00'
end_time = '2025-01-17 15:00:00'
y_zoon = 1.5
price_name = 'price'
time_name = 'trade_time'
time_fmt = '%Y-%m-%d %H:%M:%S'

kl_pd, orders = plot_orders(orders_data=orders_data,
                            market_data=market_data,
                            code='IM',
                            begin_time=begin_time,
                            end_time=end_time,
                            direction=direction,
                            time_name=time_name,
                            price_name=price_name,
                            time_fmt=time_fmt)
