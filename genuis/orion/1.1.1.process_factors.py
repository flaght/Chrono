import pdb, os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.logger import logger
from kdutils.macro2 import *
from kdutils.tactix import Tactix
from kdutils.ttimes import get_dates

from lib.pre002.processing import standardize, winsorize


def load_raw_returns(start_date=None, end_date=None):
    return_file = os.path.join(os.environ['DX_DATA_PATH'], "ret_daily_o2o.parquet")
    return_data = pd.read_parquet(return_file)
    return_data.rename(columns={'date':'trade_time','Code':'code'}, inplace=True)
    return return_data[['trade_time','code','ret','abret_300', 'abret_500', 'abret_1000', 'abret_2000', 'abret_market']]
    
def load_raw_factors1(start_date=None, end_date=None):
    factor_dirs = os.path.join(os.environ['DX_DATA_PATH'], 'signals')
    p = Path(factor_dirs)
    # 这里演示只取前两个文件
    files = [f for f in p.glob('*.fea') if 'norm' not in f.name]
    files = files[:] 
    
    print(f"找到 {len(files)} 个符合条件的因子文件")
    
    normal_wide_dfs = {} 
    original_wide_dfs = {}

    for file_path in tqdm(files, desc="Reading files"):
        df = pd.read_feather(file_path)
        factor_name = file_path.stem
        if 'trade_date' in df.columns:
            df = df.set_index('trade_date')
        elif 'trade_time' in df.columns:
            df = df.set_index('trade_time')
        df.index = pd.to_datetime(df.index)
        
        df.columns = df.columns.astype(str).str.zfill(6)
        
        ## 进行标准化处理
        f = standardize(winsorize(df))
        f[np.isnan(f)] = 0
        normal_wide_dfs[factor_name] = f
        original_wide_dfs[factor_name] = df

    logger.info("正在宽表维度进行合并 (Fast Merge)...")
    normal_combined_wide = pd.concat(normal_wide_dfs.values(), axis=1, keys=normal_wide_dfs.keys())
    normal_combined_wide[np.isnan(normal_combined_wide)] = 0
    original_combined_wide = pd.concat(original_wide_dfs.values(), axis=1, keys=original_wide_dfs.keys())
    
    print("正在转换为长表 (Stacking)...")
    normal_factors_data = normal_combined_wide.stack(dropna=False).fillna(0)
    normal_factors_data = normal_factors_data.dropna(how='all')
    normal_factors_data = normal_factors_data.loc[start_date:end_date]
    
    original_factors_data = original_combined_wide.stack()
    original_factors_data = original_factors_data.dropna(how='all')
    original_factors_data = original_factors_data.loc[start_date:end_date]
    print(f"处理完成，形状: {normal_factors_data.shape}")
    
    normal_factors_data = normal_factors_data.reset_index()
    original_factors_data = original_factors_data.reset_index()
    normal_factors_data.rename(columns={'level_1':'code','trade_date':'trade_time'}, inplace=True)
    original_factors_data.rename(columns={'level_1':'code','trade_date':'trade_time'}, inplace=True)
    return normal_factors_data, original_factors_data
    
## task_id 对应映射 source, cycle, period
def start_factors(method, task_id, source, cycle, period):
    start_date, end_date = get_dates(method)
    normal_factors_data, original_factors_data = load_raw_factors1(start_date=start_date, end_date=end_date)
    output_dirs = os.path.join(base_path, method, source, 'basic',  str(task_id))
    os.makedirs(output_dirs, exist_ok=True)
    pdb.set_trace()
    normal_factors_data.to_feather(os.path.join(output_dirs, "normal_factors.feather"))
    original_factors_data.to_feather(os.path.join(output_dirs, "original_factors.feather"))
    
### 收益率处理
def start_returns(method, task_id, source, cycle, period):
    start_date, end_date = get_dates(method)
    return_data = load_raw_returns(start_date=start_date, end_date=end_date)
    output_dirs = os.path.join(base_path, method, source, 'basic',  str(task_id))
    os.makedirs(output_dirs, exist_ok=True)
    return_data.to_feather(os.path.join(output_dirs, "return_data.feather"))
     
    

if __name__ == '__main__':
    variant = Tactix().start()
    start_returns(method=variant.method,
          source=variant.source,
          task_id=variant.task_id,
          cycle=variant.cycle,
          period=variant.period)