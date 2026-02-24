## DX AShare 切割数据 已经标准化过
import pdb, os, datetime
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.logger import logger
from kdutils.macro2 import *
from kdutils.tactix import Tactix
from kdutils.ttimes import get_dates

def process_normal(method, task_id, source, cycle, period):
    ## 加载标准化的因子
    base_dirs = os.path.join(base_path, method, source, 'basic', str(task_id))
    factor_data = pd.read_feather(os.path.join(base_dirs, "normal_factors.feather"))
    return_data =  pd.read_feather(os.path.join(base_dirs, "return_data.feather"))
    factor_data['trade_time'] = pd.to_datetime(factor_data['trade_time'])
    return_data['trade_time'] = pd.to_datetime(return_data['trade_time'])
    
    total_data = factor_data.merge(return_data, on=['trade_time','code'])
    ## 对齐数据
    total_data_unstack = total_data.set_index(['trade_time','code']).unstack()
    total_data1 = total_data_unstack.fillna(0).stack(dropna=False).fillna(0)
    total_data1 = total_data1.sort_index()
    total_data1 = total_data1.reset_index()
    
    ## 数据切割
    total_data1['trade_time'] = pd.to_datetime(total_data1['trade_time']).dt.strftime('%Y-%m-%d')
    ### 切割时间
    times = total_data1['trade_time'].unique().tolist()
    
    len1 = round(len(times) * 0.6)  # 60%部分
    len2 = round(len(times) * 0.2)  # 25%部分
    len3 = len(times) - len1 - len2
    
    train_data = total_data1[total_data1['trade_time'].isin(times[:len1])]
    val_data = total_data1[total_data1['trade_time'].isin(times[len1:len1 +
                                                                  len2])]
    test_data = total_data1[total_data1['trade_time'].isin(times[len1 +
                                                                   len2:])]
    
    target_dir = os.path.join(base_path, method, source, 'rl', str(task_id))
    os.makedirs(target_dir, exist_ok=True)
    pdb.set_trace()
    train_data.reset_index(drop=True).to_feather(os.path.join(target_dir, "train_data.feather"))
    val_data.reset_index(drop=True).to_feather(os.path.join(target_dir, "val_data.feather"))
    test_data.reset_index(drop=True).to_feather(os.path.join(target_dir, "test_data.feather"))
    
    
def process_original(method, task_id):
    base_dirs = os.path.join(base_path, method, TASK_MAPPING[task_id]['source'], 'basic', str(task_id))
    factor_data = pd.read_feather(os.path.join(base_dirs, "original_factors.feather"))
    return_data =  pd.read_feather(os.path.join(base_dirs, "return_data.feather"))
    factor_data['trade_time'] = pd.to_datetime(factor_data['trade_time'])
    return_data['trade_time'] = pd.to_datetime(return_data['trade_time'])
    
    total_data1 = factor_data.merge(return_data, on=['trade_time','code'])
    ## 数据切割
    total_data1['trade_time'] = pd.to_datetime(total_data1['trade_time']).dt.strftime('%Y-%m-%d')
    ### 切割时间
    times = total_data1['trade_time'].unique().tolist()
    
    len1 = round(len(times) * 0.6)  # 60%部分
    len2 = round(len(times) * 0.2)  # 25%部分
    len3 = len(times) - len1 - len2
    
    train_data = total_data1[total_data1['trade_time'].isin(times[:len1])]
    val_data = total_data1[total_data1['trade_time'].isin(times[len1:len1 +
                                                                  len2])]
    test_data = total_data1[total_data1['trade_time'].isin(times[len1 +
                                                                   len2:])]
    
    target_dir = os.path.join(base_path, method, TASK_MAPPING[task_id]['source'], 'base', str(task_id))
    os.makedirs(target_dir, exist_ok=True)
    
    train_factors = train_data[factor_data.columns]
    val_factors = val_data[factor_data.columns]
    test_factors = test_data[factor_data.columns]
    
    train_return = train_data[return_data.columns]
    val_return = val_data[return_data.columns]
    test_return = test_data[return_data.columns]
    pdb.set_trace()
    train_factors.reset_index(drop=True).to_feather(os.path.join(target_dir, "train_data.feather"))
    val_factors.reset_index(drop=True).to_feather(os.path.join(target_dir, "val_data.feather"))
    test_factors.reset_index(drop=True).to_feather(os.path.join(target_dir, "test_data.feather"))
    
    
    train_return.reset_index(drop=True).to_feather(os.path.join(target_dir, "train_return.feather"))
    val_return.reset_index(drop=True).to_feather(os.path.join(target_dir, "val_return.feather"))
    test_return.reset_index(drop=True).to_feather(os.path.join(target_dir, "test_return.feather"))
    

if __name__ == '__main__':
    variant = Tactix().start()
    process_original(method=variant.method,
          task_id=variant.task_id)