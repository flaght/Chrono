import pdb, os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from lib.composite.loader import DataLoader
from lib.composite.cleaner import DataCleaner
from lib.composite.feature import Featurer
from kdutils.tactix import Tactix
from kdutils.macro2 import base_path

from lib.aux001 import fetch_temp_returns

def create_data(method, instruments, task_id, period, name):
    pdb.set_trace()
    train_data,val_data, test_data = DataLoader().load_from_project(method=method, task_id=task_id, 
                                    instruments=instruments, 
                                    period=period, name=name,
                                    features=[])
    
    train_return = fetch_temp_returns(method=method,
                                    instruments=instruments,
                                    category='returns',
                                    datasets=['train'])
    val_return = fetch_temp_returns(method=method,
                                    instruments=instruments,
                                    category='returns',
                                    datasets=['val'])
    
    test_return = fetch_temp_returns(method=method,
                                    instruments=instruments,
                                    category='returns',
                                    datasets=['test'])
    
    ### 加载regmie
    output_dirs = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl', 'regime')
    min_regime_factors = pd.read_feather(os.path.join(output_dirs, "min.feather"))
    daily_regime_factors = pd.read_feather(os.path.join(output_dirs, "daily.feather"))
    pdb.set_trace()
    train_data = train_data.merge(
        train_return[['trade_time','code','nxt1_ret_1h']], on=['trade_time','code'])
    train_data = train_data.merge(min_regime_factors, on=['trade_time','code'],how='left')
    
    val_data = val_data.merge(
        val_return[['trade_time','code','nxt1_ret_1h']], on=['trade_time','code'])
    
    val_data = val_data.merge(min_regime_factors, on=['trade_time','code'],how='left')
    
    
    test_data = test_data.merge(
        test_return[['trade_time','code','nxt1_ret_1h']], on=['trade_time','code'])
    test_data = test_data.merge(min_regime_factors, on=['trade_time','code'],how='left')
    
    
    train_data['trade_time'] = pd.to_datetime(train_data['trade_time'])
    val_data['trade_time'] = pd.to_datetime(val_data['trade_time'])
    test_data['trade_time'] = pd.to_datetime(test_data['trade_time'])
    
    ## 前值填充
    daily_regime_factors = daily_regime_factors.set_index(
        ['trade_time','code']).unstack().fillna(method='ffill').stack().reset_index()
    daily_regime_factors['trade_time'] = pd.to_datetime(daily_regime_factors['trade_time'])
    daily_regime_factors['trade_time'] = daily_regime_factors['trade_time'].dt.normalize()
    
    train_valid_dates = train_data['trade_time'].dt.normalize().unique()
    train_regime_daily = daily_regime_factors[daily_regime_factors['trade_time'].isin(train_valid_dates)].copy()
    
    val_valid_dates = val_data['trade_time'].dt.normalize().unique()
    val_regime_daily = daily_regime_factors[daily_regime_factors['trade_time'].isin(val_valid_dates)].copy()
    
    
    test_valid_dates = test_data['trade_time'].dt.normalize().unique()
    test_regime_daily = daily_regime_factors[daily_regime_factors['trade_time'].isin(test_valid_dates)].copy()
    
    pdb.set_trace()
    
    output_dirs = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl', 'data')
    os.makedirs(output_dirs, exist_ok=True)
    pdb.set_trace()
    train_data.reset_index(drop=True).to_feather(os.path.join(output_dirs, "train_data.feather"))
    val_data.reset_index(drop=True).to_feather(os.path.join(output_dirs, "val_data.feather"))
    test_data.reset_index(drop=True).to_feather(os.path.join(output_dirs, "test_data.feather"))
    
    train_regime_daily.reset_index(drop=True).to_feather(os.path.join(output_dirs, "train_regime.feather"))
    val_regime_daily.reset_index(drop=True).to_feather(os.path.join(output_dirs, "val_regime.feather"))
    test_regime_daily.reset_index(drop=True).to_feather(os.path.join(output_dirs, "test_regime.feather"))
    
    

if __name__ == '__main__':
    variant = Tactix().start()
    create_data(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name)