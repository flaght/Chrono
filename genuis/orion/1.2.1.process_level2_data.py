import pdb, os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from alphacopilot.api.data import RetrievalAPI
from kdutils.macro2 import *
from kdutils.tactix import Tactix
from kdutils.ttimes import get_dates

def load_raw_data(filename, method, task_id, start_date=None, end_date=None):
    pdb.set_trace()
    filename = os.path.join(filename)
    factors_data = pd.read_parquet(filename)
    factors_data = factors_data.rename(columns={'Code': 'symbol'})
    factors_data['minTime'] = factors_data['minTime'].astype(str).str.zfill(6)
    datetime_str = factors_data['date'].astype(
        str) + factors_data['minTime'].astype(str)
    factors_data['trade_time'] = pd.to_datetime(datetime_str,
                                                format='%Y%m%d%H%M%S')
    factors_data = factors_data.drop(columns=['date', 'minTime'])
    regex_pattern = r'^([A-Za-z]+)'
    factors_data['code'] = factors_data['symbol'].str.extract(regex_pattern)
    factors_data = factors_data.set_index(
        'trade_time').loc[start_date:end_date].reset_index()
    
    factors_data['trade_time'] = pd.to_datetime(
        factors_data['trade_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    algin_data = RetrievalAPI.get_algin_factors(
        begin_date=start_date, end_date=end_date, 
        codes=factors_data['code'].unique().tolist())
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    factors_data['trade_date'] = factors_data['trade_time'].dt.normalize()
    algin_data['trade_date'] = pd.to_datetime(algin_data['trade_date'])
    dominant_contracts_lookup = algin_data[['trade_date', 'symbol']].drop_duplicates().copy()
    merged_data = pd.merge(factors_data,dominant_contracts_lookup,on=['trade_date', 'symbol'],how='inner')
    factors_data = merged_data.copy()
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    output_dir = os.path.join(base_path, method, TASK_MAPPING[task_id]['source'], 'basic', task_id)
    os.makedirs(output_dir, exist_ok=True)
    factors_data.to_feather(os.path.join(output_dir, "original_factors.feather"))
    
    
    
    
def start(method, task_id, filename):
    start_date, end_date = get_dates(method)
    load_raw_data(filename=filename, method=method, task_id=task_id,
                  start_date=start_date, end_date=end_date)
    

if __name__ == '__main__':
    variant = Tactix().start()
    start(method=variant.method,
          task_id=variant.task_id,
          filename=variant.filename)