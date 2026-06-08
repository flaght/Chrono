## 构建fake因子 和已有因子合成

import pdb
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression  
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



features = ["MSUM(120,MDEMA(90,MCPS(90,'high')))", 
            "MDEMA(120,MCPS(120,WMA(90,'twap')))",           
                             "MMAX(120,MDEMA(60,MCPS(90,'low')))", "MMASSI(120,MPRO(60,MVHF(10,'money')),MAPOSITIVE(10,'twap'))",         
                             "MDEMA(120,MCPS(120,MADecay(60,'twap')))", "MT3(120,MCPS(30,'close'))",                                      
                             "MA(60,RSI(120,MCPS(120,MA(60,'twap'))))", "MT3(120,MCPS(60,'high'))",                                       
                             "DELTA(90,MMIN(15,MHMA(90,DELTA(90,'close'))))/MDIFF(90,'close')", "MCPS(120,MT3(90,MMaxDiff(120,'twap')))", 
                             "MADecay(5,MMASSI(120,MT3(5,'corr_vwap_bid_size_0'),'twap'))",                                               
                             "MMeanRes(120,'corr_money_bid_size_0','smart_tick_in_pct')", "WMA(30,MMedian(90,'smart_tick_in_pct'))",      
                             "MMAX(15,MDPO(240,EMA(90,'smart_money_in_pct')))", "RSI(120,MCPS(120,EMA(120,'close')))",                    
                             "MSUM(120,MDEMA(90,MCPS(90,'low')))", "MDIFF(90,MMeanRes(120,'corr_money_bid_size_0','smart_tick_in_pct'))", 
                             "MSUM(5,MADecay(10,MMedian(90,'smart_tick_in_pct')))", "MMedian(90,MADecay(10,MT3(5,'smart_tick_in_pct')))"] 

name = "CANARY_GOOD_20"
train_tmp = train_data[['trade_time', 'code'] +  features + [name] + ['nxt1_ret']].copy().dropna()
val_tmp = val_data[['trade_time', 'code'] +  features + [name] + ['nxt1_ret']].copy().dropna()

train_tmp[features] = train_tmp[features].replace([np.inf, -np.inf], np.nan).dropna()
val_tmp[features] = val_tmp[features].replace([np.inf, -np.inf], np.nan).dropna()

X = train_tmp[features].astype(float).values                 # (N,19)
y = train_tmp[name].astype(float).values  # (N,)

model = LinearRegression(fit_intercept=True)  # 建议 True
model.fit(X, y)

val_X = val_tmp[features].astype(float).values  
val_y = val_tmp[name].astype(float).values
resid = val_y - model.predict(val_X)
val_tmp['resid'] = resid

evaluate1 = FactorEvaluate1(factor_data=val_tmp.copy(),
                                factor_name=name,
                                ret_name='nxt1_ret',
                                roll_win=15,
                                fee=0.000,
                                scale_method='raw',
                                expression='expression',
                                resampling_win=15)
stats_dt1 = evaluate1.run()


evaluate2 = FactorEvaluate1(factor_data=val_tmp.copy(),
                                factor_name="resid",
                                ret_name='nxt1_ret',
                                roll_win=15,
                                fee=0.000,
                                scale_method='raw',
                                expression='expression',
                                resampling_win=15)
stats_dt2 = evaluate2.run()
pdb.set_trace()
print('-->')


# dt = pd.concat([train_data, val_data],axis=0).sort_values(by=['trade_time','code']).dropna()

# tmp = dt[['trade_time', 'code'] +  features + ["MRANK(5, MSKEW(30, 'pct_change'))"] + ['nxt1_ret']].copy()
# tmp[features] = tmp[features].replace([np.inf, -np.inf], np.nan).dropna()


# ys = dt["MRANK(5, MSKEW(30, 'pct_change'))"].astype(float).values.reshape(-1,1)
# Xs = dt["MSUM(120,MDEMA(90,MCPS(90,'high')))"].astype(float).values.reshape(-1,1)
# pdb.set_trace()
# model = LinearRegression(fit_intercept=False)
# model.fit(Xs, ys)
# resid = ys - model.predict(Xs)
# dt["resid"] = resid

# pdb.set_trace()
# X = tmp[features].astype(float).values                 # (N,19)
# y = tmp["MRANK(5, MSKEW(30, 'pct_change'))"].astype(float).values  # (N,)

# model = LinearRegression(fit_intercept=True)  # 建议 True
# model.fit(X, y)

# resid = y - model.predict(X)

# tmp['resid'] = resid

# evaluate1 = FactorEvaluate1(factor_data=tmp.copy(),
#                                 factor_name="MRANK(5, MSKEW(30, 'pct_change'))",
#                                 ret_name='nxt1_ret',
#                                 roll_win=15,
#                                 fee=0.000,
#                                 scale_method='raw',
#                                 expression='expression',
#                                 resampling_win=15)
# stats_dt1 = evaluate1.run()

# evaluate2 = FactorEvaluate1(factor_data=tmp.copy(),
#                                 factor_name="resid",
#                                 ret_name='nxt1_ret',
#                                 roll_win=15,
#                                 fee=0.000,
#                                 scale_method='raw',
#                                 expression='expression',
#                                 resampling_win=15)
# stats_dt2 = evaluate2.run()
# pdb.set_trace()

# print('-->')
# # mappings = {}
# # for s in ['CANARY_GOOD_10','CANARY_GOOD_20', 'CANARY_GOOD_60']:
# #     mpp = []
# #     for f in features:
# #         corr_values = dt[s].rolling(window=15).corr(dt[f])
# #         mpp.append({'name': f, 'value':corr_values.dropna().mean()})
    
# #     mappings[s] = mpp
# # pdb.set_trace()
# # print('-->')