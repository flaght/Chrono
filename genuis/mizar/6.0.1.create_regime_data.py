import pdb, os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from lib.composite.loader import DataLoader
from lib.composite.cleaner import DataCleaner
from lib.composite.feature import Featurer
from kdutils.tactix import Tactix
from kdutils.macro2 import base_path

from lib.aux001 import fetch_temp_data

### 计算分钟频regmie， 计算日频regmie
def create_regmie_data(method, instruments, task_id, period):
    eps = 1e-8
    total_factors = fetch_temp_data(method=method,
                                    task_id=task_id,
                                    instruments=instruments,
                                    datasets=['train', 'val','test'])
    total_factors['trade_time'] = pd.to_datetime(total_factors['trade_time'])
    pdb.set_trace()
    total_factors = total_factors.set_index(['trade_time','code'])
    total_data = total_factors.unstack()
    pct = total_data['pct_change'].fillna(0)
    high = total_data['high']
    low = total_data['low']
    close = total_data['close']
    
    rv_5 = np.sqrt((pct ** 2).rolling(window=5, min_periods=1).sum())
    rv_15 = np.sqrt((pct ** 2).rolling(window=15, min_periods=1).sum())
    rv_60 = np.sqrt((pct ** 2).rolling(window=60, min_periods=1).sum())
    
    vol_ratio_5_60 = rv_5 / (rv_60 + eps)

    # 计算 zvol
    rv_15_mean = rv_15.rolling(240, min_periods=1).mean()
    rv_15_std = rv_15.rolling(240, min_periods=1).std()
    zvol_15 = (rv_15 - rv_15_mean) / (rv_15_std + eps)

    # 计算 range
    range_15 = (high.rolling(15, min_periods=1).max() - low.rolling(15, min_periods=1).min()) / (close + eps)
    
    min_factors = pd.concat([rv_5, rv_15, rv_60, vol_ratio_5_60, zvol_15, 
                              range_15], axis=1, keys=['rv_5', 'rv_15', 'rv_60', 
                                                       'vol_ratio_5_60', 'zvol_15', 
                                                       'range_15'])
    
    ## 将基础数据降频 
    agg_rules = {
        'open':  'first', 
        'high':  'max', 
        'low':   'min', 
        'close': 'last',
        'volume': 'sum'
    }
    pdb.set_trace()
    
    total_factors['day_rv'] = total_factors['pct_change'] ** 2
    trade_dates = total_factors.index.get_level_values('trade_time').normalize()
    daily_var = total_factors['day_rv'].groupby([trade_dates, 'code']).sum()
    daily_vol = np.sqrt(daily_var)
    
    
    
    daily_data = total_factors[['open','high','low','close','volume']].groupby(level='code').resample('1D', level='trade_time').agg(agg_rules).dropna()
    daily_data = pd.concat([daily_data.swaplevel('trade_time','code'),daily_vol],axis=1).unstack()
    
    pdb.set_trace()
    close = daily_data['close']
    high = daily_data['high']
    low = daily_data['low']
    open = daily_data['open']
    day_rv = daily_data['day_rv']
    
    hist_vol_5d =  (close / close.shift(1)).rolling(window=5).std() * np.sqrt(252)
    gk_5d = np.sqrt(
        ((0.5 * np.log(high/ low) ** 2)) - (
            (2 * np.log(2) - 1) * np.log(close / open) ** 2).rolling(window=5).mean())  * np.sqrt(252)

    rv_20d = day_rv.rolling(20, min_periods=1).mean()
    
    daily_factors = pd.concat([hist_vol_5d, gk_5d, rv_20d], axis=1, keys=['hist_vol_5d', 'gk_5d', 'rv_20d'])
    daily_factors = daily_factors.shift(1)
    
    output_dirs = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl', 'regime')
    os.makedirs(output_dirs)
    min_factors.stack().reset_index().to_feather(os.path.join(output_dirs, "min.feather"))
    daily_factors.stack().reset_index().to_feather(os.path.join(output_dirs, "daily.feather"))
    

if __name__ == '__main__':
    variant = Tactix().start()
    create_regmie_data(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period)