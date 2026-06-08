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
