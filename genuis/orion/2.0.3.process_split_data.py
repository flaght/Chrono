### Crypto 切割数据
import pdb, os, datetime
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.tactix import Tactix
from kdutils.macro2 import base_path


## 切割因子数据+原始数据，不做标准化
def split_factors(method, period, source):
    ## 加载因子数据
    pdb.set_trace()
    dirs = os.path.join(base_path, method, 'derivative', period, source)
    file_path = Path(dirs)
    factors_files = [x for x in file_path.glob('*factors*') if x.is_file()]
    res = []
    for file in factors_files:
        data = pd.read_feather(file)
        res.append(data)
    factors_data = pd.concat(res, axis=0)

    ### 加载收益率数据
    returns_data = pd.read_feather(os.path.join(dirs, "returns_data.feather"))

    ### 加载基础数据
    dirs = os.path.join(base_path, method, 'basic', period, source)
    file_name = os.path.join(dirs, "raw_basic.feather")
    raw_basic_data = pd.read_feather(file_name)
    raw_basic_data[
        's_vwap'] = raw_basic_data['s_value'] / raw_basic_data['s_vol']
    raw_basic_data[
        'f_vwap'] = raw_basic_data['f_value'] / raw_basic_data['f_vol']
    pdb.set_trace()
    ## 数据合并
    factors_data = factors_data.merge(returns_data,
                                      on=['trade_time', 'code'
                                          ]).merge(raw_basic_data,
                                                   on=['trade_time', 'code'])

    ## 删除列全部为nan
    nan_columns = factors_data.columns[factors_data.isna().all()]
    factors_data = factors_data.drop(nan_columns, axis=1)
    ## 删除行全部为nan
    factors_data = factors_data.dropna().reset_index(drop=True)

    factors_data['trade_time'] = pd.to_datetime(
        factors_data['trade_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    factors_data = factors_data.sort_values(by=['trade_time', 'code'])

    ### 切割时间
    times = factors_data['trade_time'].unique().tolist()

    len1 = round(len(times) * 0.6)  # 60%部分
    len2 = round(len(times) * 0.2)  # 25%部分
    len3 = len(times) - len1 - len2

    train_data = factors_data[factors_data['trade_time'].isin(times[:len1])]
    val_data = factors_data[factors_data['trade_time'].isin(times[len1:len1 +
                                                                  len2])]
    test_data = factors_data[factors_data['trade_time'].isin(times[len1 +
                                                                   len2:])]

    returns_columns = factors_data.filter(regex="^nxt1").columns.to_list()
    factors_columns = factors_data.filter(regex="^ak").columns.to_list()
    other_columns = [
        f for f in factors_data.columns
        if f not in returns_columns + factors_columns + ['trade_time', 'code']
    ]
    
    target_dir = os.path.join(base_path, method, 'base', period, source)
    os.makedirs(target_dir, exist_ok=True)

    train_features_data = train_data[['trade_time', 'code'] + factors_columns +
                                     other_columns]
    val_features_data = val_data[['trade_time', 'code'] + factors_columns +
                                 other_columns]
    test_features_data = test_data[['trade_time', 'code'] + factors_columns +
                                   other_columns]

    train_returns_data = train_data[['trade_time', 'code'] + returns_columns]
    val_returns_data = val_data[['trade_time', 'code'] + returns_columns]
    test_returns_data = test_data[['trade_time', 'code'] + returns_columns]

    train_features_data.reset_index(drop=True).to_feather(
        os.path.join(target_dir, "train_data.feather"))
    val_features_data.reset_index(drop=True).to_feather(
        os.path.join(target_dir, "val_data.feather"))
    test_features_data.reset_index(drop=True).to_feather(
        os.path.join(target_dir, "test_data.feather"))

    train_returns_data.reset_index(drop=True).to_feather(
        os.path.join(target_dir, "train_return.feather"))
    val_returns_data.reset_index(drop=True).to_feather(
        os.path.join(target_dir, "val_return.feather"))
    test_returns_data.reset_index(drop=True).to_feather(
        os.path.join(target_dir, "test_return.feather"))


if __name__ == '__main__':
    variant = Tactix().start()
    split_factors(method=variant.method,
                  period=variant.period,
                  source=variant.source)
