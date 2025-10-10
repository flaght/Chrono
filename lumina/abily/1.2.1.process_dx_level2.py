##处理DX的 level2数据
import pdb, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *


def merge(method):

    def spliter(factors_data, begin_date, end_date, name='train'):
        factors_data = factors_data.set_index(
            'trade_time').loc[begin_date:end_date].reset_index()
        pdb.set_trace()
        codes = factors_data.code.unique().tolist()
        mapping = {'IM': 'ims', 'IC': 'ics', 'IF': 'ifs', 'IH': 'ihs'}
        #dirs = os.path.join(base_path, method, instruments, 'level2')
        #if not os.path.exists(dirs):
        #    os.makedirs(dirs)
        for code in codes:
            instruments = mapping[code]
            dirs = os.path.join(base_path, method, instruments, 'level2')
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            fd = factors_data[factors_data.code.isin([code])]
            filename = os.path.join(dirs, f"{name}_data.feather")
            print(filename)
            fd.reset_index(drop=True).to_feather(filename)

    filename = os.path.join('/workspace/worker/feature_future_1min_df.parquet')
    factors_data = pd.read_parquet(filename)
    factors_data = factors_data.reset_index()
    factors_data = factors_data.rename(columns={'Code': 'symbol'})
    factors_data['minTime'] = factors_data['minTime'].astype(str).str.zfill(6)
    datetime_str = factors_data['date'].astype(
        str) + factors_data['minTime'].astype(str)
    factors_data['trade_time'] = pd.to_datetime(datetime_str,
                                                format='%Y%m%d%H%M%S')
    factors_data = factors_data.drop(columns=['date', 'minTime'])
    regex_pattern = r'^([A-Za-z]+)'
    factors_data['code'] = factors_data['symbol'].str.extract(regex_pattern)
    pdb.set_trace()
    ### 训练集
    train_begin_date = "2022-07-25 09:30:00"
    train_end_date = "2024-05-29 13:22:00"
    spliter(factors_data=factors_data.copy(),
            begin_date=train_begin_date,
            end_date=train_end_date,
            name='train')
    ### 校验集
    val_begin_date = "2024-05-29 13:23:00"
    val_end_date = "2024-12-05 10:15:00"
    spliter(factors_data=factors_data.copy(),
            begin_date=val_begin_date,
            end_date=val_end_date,
            name='val')
    ### 测试集
    test_begin_date = "2024-12-05 10:16:00"
    test_end_date = "2025-04-10 15:00:00 "
    spliter(factors_data=factors_data.copy(),
            begin_date=test_begin_date,
            end_date=test_end_date,
            name='test')


merge(method='aicso0')
