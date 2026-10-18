### 构建fake 因子测试

import pdb
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()

from lib.iux001 import fetch_data, aggregation_data,merging_data1
from lib.aux001 import calc_expression
from lib.cux001 import FactorEvaluate1

def add_canary(df, rho=0.6, seed=42):
    rng = np.random.default_rng(seed)
    y = df['nxt1_ret'].astype(float).to_numpy()

    mu = np.nanmean(y)
    sd = np.nanstd(y) + 1e-8

    zy = (y - mu) / sd
    e = rng.normal(size=len(df))
    x = rho * zy + np.sqrt(1 - rho**2) * e 
    df[f"CANARY_GOOD_{int(rho*100):02d}"] = x
    return df
    
    

ret_name = "nxt1_ret_15h"
train_data = pd.read_feather("./records/cicso1/ims/temp/model/200037/15/rl/data/train_data.feather")
val_data = pd.read_feather("./records/cicso1/ims/temp/model/200037/15/rl/data/val_data.feather")

train_data.rename(columns={ret_name: "nxt1_ret"},
                      inplace=True)

val_data.rename(columns={ret_name: "nxt1_ret"},
                    inplace=True)

train_data = add_canary(train_data, rho=0.6)
val_data = add_canary(val_data, rho=0.6)

train_data = add_canary(train_data, rho=0.2)
val_data = add_canary(val_data, rho=0.2)

train_data = add_canary(train_data, rho=0.1)
val_data = add_canary(val_data, rho=0.1)

pdb.set_trace()
print('-->')
dt = pd.concat([train_data, val_data],axis=0).sort_values(by=['trade_time','code'])
evaluate1 = FactorEvaluate1(factor_data=dt.copy(),
                                factor_name='CANARY_GOOD_60',
                                ret_name='nxt1_ret',
                                roll_win=15,
                                fee=0.000,
                                scale_method='roll_zscore',
                                expression='expression',
                                resampling_win=15)
stats_dt1 = evaluate1.run()
pdb.set_trace()

evaluate2 = FactorEvaluate1(factor_data=dt.copy(),
                                factor_name='CANARY_GOOD_10',
                                ret_name='nxt1_ret',
                                roll_win=15,
                                fee=0.000,
                                scale_method='roll_zscore',
                                expression='expression',
                                resampling_win=15)
stats_dt2 = evaluate2.run()


evaluate3 = FactorEvaluate1(factor_data=dt.copy(),
                                factor_name='CANARY_GOOD_20',
                                ret_name='nxt1_ret',
                                roll_win=15,
                                fee=0.000,
                                scale_method='roll_zscore',
                                expression='expression',
                                resampling_win=15)
stats_dt3 = evaluate3.run()



# evaluate2 = FactorEvaluate1(factor_data=dt.copy(),
#                                 factor_name='nxt1_ret_15h',
#                                 ret_name='nxt1_ret',
#                                 roll_win=15,
#                                 fee=0.000,
#                                 scale_method='roll_zscore',
#                                 expression='expression',
#                                 resampling_win=15)
# stats_dt2 = evaluate2.run()
pdb.set_trace()

train_data.to_feather("./records/cicso1/ims/temp/model/200037/15/rl/data/train_data_fake.feather")
val_data.to_feather("./records/cicso1/ims/temp/model/200037/15/rl/data/val_data_fake.feather")