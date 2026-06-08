import os, sys, pdb, re
import pandas as pd

sys.path.insert(0, os.path.abspath('../'))

from lumina.formual.impulse import Impulse

#dependencies = ['tc008_5_10_1', 'rv010_10_15_1_2']
dependencies = [
    'db002_5_10_1', 'tc008_5_10_1', 'rv010_10_15_1_2', 'oi011_5_10_1',
    'gd002_5_10_1', 'tv001_5_10', 'rv005_5_10_1_2', 'rv006_10_15_1_2',
    'rv013_75_5_10_0', 'rv006_5_10_1_2', 'cj003_5_10_1', 'oi023_5_10_1',
    'rv010_5_10_1_2', 'ixy014_5_10_1', 'oi042_10_15_1', 'tc013_5_10_1',
    'iv012_5_10_1', 'rv005_10_15_1_2', 'cj011_5_10_1', 'tc003_5_10_1',
    'oi039_5_10_1', 'tv017_5_10_1', 'rv005_10_15_0_2', 'oi037_5_10_1',
    'tc015_5_10_1', 'tc006_5_10_0', 'oi042_5_10_1', 'oi004_5_10_1',
    'tc006_5_10_1', 'oi027_5_10_1', 'rv005_5_10_0_2'
]

cols = ['open', 'high', 'low', 'close', 'volume', 'value', 'openint']
bar_data = pd.read_feather("bar_data.feather")
bar_data = bar_data.loc[40:142]
pdb.set_trace()
bar_data.rename(columns={
    'datetime': 'trade_time',
    'symbol': 'code'
},
                inplace=True)
bar_data.sort_values(by=['trade_time', 'code'], inplace=True)
bar_data = bar_data.set_index(['trade_time', 'code'])[cols]
res = {}
pdb.set_trace()
for col in cols:
    res[col] = bar_data[col].unstack()
factors_data = Impulse(dependencies).batch(res)
pdb.set_trace()
print(bar_data)
