### Crypto 切割数据
import pdb, os, datetime
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.tactix import Tactix
from kdutils.macro2 import base_path

## 合并数据


def merge(min_data, daily_data):
    min_data['trade_time'] = pd.to_datetime(min_data['trade_time'])
    daily_data['trade_time'] = pd.to_datetime(daily_data['trade_time'])

    min_data['join_date'] = min_data['trade_time'].dt.normalize()
    daily_data['trade_time'] = daily_data['trade_time'].dt.normalize()
    merged_data = pd.merge(
        min_data,
        daily_data,
        left_on=['join_date', 'code'],  # 左表键
        right_on=['trade_time', 'code'],  # 右表键
        how='left',  # 保证分钟线行数不变
        suffixes=('', '_daily')  # 如果有重名列，右表加后缀
    )
    cols_to_drop = ['join_date']
    if 'trade_time_daily' in merged_data.columns:
        cols_to_drop.append('trade_time_daily')

    merged_data = merged_data.drop(columns=['trade_time_y'],
                                   errors='ignore')  # 默认后缀是_x, _y
    merged_data = merged_data.drop(columns=['join_date'], errors='ignore')
    print(f"合并前行数: {len(min_data)}, 合并后行数: {len(merged_data)}")
    return merged_data


## 切割因子数据+原始数据，不做标准化
def split_factors(method, task_id):
    ## 加载因子数据
    pdb.set_trace()
    dirs = os.path.join(base_path, method, 'derivative', task_id)
    file_path = Path(dirs)
    factors_files = [x for x in file_path.glob('*factors*') if x.is_file()]
    res = []
    factors_data = None
    for file in factors_files:
        print(file)
        data = pd.read_feather(file)
        data = data.sort_values(by=['trade_time', 'code'])
        data = data.set_index([
            'trade_time', 'code'
        ]).unstack().fillna(method='ffill').stack().reset_index().sort_values(
            by=['trade_time', 'code'])
        if factors_data is None:
            factors_data = data
        else:
            factors_data = factors_data.merge(data, on=['trade_time', 'code'])
        # res.append(data)
    # pdb.set_trace()
    # factors_data = pd.concat(res, axis=0)
    ### 加载收益率数据
    returns_data = pd.read_feather(os.path.join(dirs, "returns_data.feather"))

    ### 加载基础数据
    dirs = os.path.join(base_path, method, 'basic', task_id)
    file_name = os.path.join(dirs, "raw_basic.feather")
    raw_basic_data = pd.read_feather(file_name)
    raw_basic_data['vwap'] = raw_basic_data['value'] / raw_basic_data['volume']
    
    returns_data = returns_data.sort_values(by=['trade_time','code'])
    raw_basic_data = raw_basic_data.sort_values(by=['trade_time','code'])
    factors_data = factors_data.sort_values(by=['trade_time','code'])
    pdb.set_trace()
    ## 数据合并
    factors_data = factors_data.merge(returns_data,
                                      on=['trade_time', 'code'
                                          ]).merge(raw_basic_data,
                                                   on=['trade_time', 'code'])
    ## 删除列全部为nan
    pdb.set_trace()
    nan_columns = factors_data.columns[factors_data.isna().all()]
    factors_data = factors_data.drop(nan_columns, axis=1)
    ## 删除行全部为nan
    factors_data = factors_data.dropna().reset_index(drop=True)

    factors_data['trade_time'] = pd.to_datetime(
        factors_data['trade_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    factors_data = factors_data.sort_values(by=['trade_time', 'code'])

    ### 切割时间
    times = factors_data['trade_time'].unique().tolist()

    len1 = round(len(times) * 0.65)  # 60%部分
    len2 = round(len(times) * 0.25)  # 25%部分
    len3 = len(times) - len1 - len2

    train_data = factors_data[factors_data['trade_time'].isin(times[:len1])]
    val_data = factors_data[factors_data['trade_time'].isin(times[len1:len1 +
                                                                  len2])]
    test_data = factors_data[factors_data['trade_time'].isin(times[len1 +
                                                                   len2:])]

    returns_columns = factors_data.filter(regex="^nxt1").columns.to_list()
    factors_columns = [
        'xy001_2_3_1', 'xy001_5_10_0', 'xy001_5_10_1', 'xy001_2_3_0',
        'xy002_2_3_1', 'xy002_5_10_0', 'xy002_5_10_1', 'xy002_2_3_0',
        'xy003_2_3_1', 'xy003_5_10_0', 'xy003_5_10_1', 'xy003_2_3_0',
        'xy004_2_3_1', 'xy004_5_10_0', 'xy004_5_10_1', 'xy004_2_3_0',
        'xy005_2_3_1', 'xy005_5_10_0', 'xy005_5_10_1', 'xy005_2_3_0',
        'ixy001_2_3_1', 'ixy001_2_3_0', 'ixy001_5_10_1', 'ixy001_5_10_0',
        'ixy002_2_3_1', 'ixy002_2_3_0', 'ixy002_5_10_1', 'ixy002_5_10_0',
        'ixy003_2_3_1', 'ixy003_2_3_0', 'ixy003_5_10_1', 'ixy003_5_10_0',
        'ixy004_2_3_1', 'ixy004_2_3_0', 'ixy004_5_10_1', 'ixy004_5_10_0',
        'ixy005_2_3_1', 'ixy005_2_3_0', 'ixy005_5_10_1', 'ixy005_5_10_0',
        'ixy006_2_3_1', 'ixy006_2_3_0', 'ixy006_5_10_1', 'ixy006_5_10_0',
        'ixy007_2_3_1', 'ixy007_2_3_0', 'ixy007_5_10_1', 'ixy007_5_10_0',
        'ixy008_2_3_1', 'ixy008_2_3_0', 'ixy008_5_10_1', 'ixy008_5_10_0',
        'ixy009_5_10_1', 'ixy009_5_10_0', 'ixy010_2_3_1', 'ixy010_2_3_0',
        'ixy010_5_10_1', 'ixy010_5_10_0', 'db001_2_3_1', 'db001_5_10_0',
        'db001_5_10_1', 'db001_2_3_0', 'db002_5_10_0', 'db002_5_10_1',
        'db002_2_3_0', 'db003_2_3_1', 'db003_5_10_0', 'db003_5_10_1',
        'db003_2_3_0', 'db004_2_3_1', 'db004_5_10_0', 'db004_5_10_1',
        'db004_2_3_0', 'db005_2_3_1', 'db005_5_10_0', 'db005_5_10_1',
        'db005_2_3_0', 'db006_2_3_1', 'db006_5_10_0', 'db006_5_10_1',
        'db006_2_3_0', 'db007_2_3_1', 'db007_5_10_0', 'db007_5_10_1',
        'db007_2_3_0', 'cj002_2_3_1', 'cj002_5_10_0', 'cj002_5_10_1',
        'cj002_2_3_0', 'cj003_2_3_1', 'cj003_5_10_0', 'cj003_5_10_1',
        'cj003_2_3_0', 'cj006_2_3_1', 'cj006_5_10_0', 'cj006_5_10_1',
        'cj006_2_3_0', 'cj007_2_3_1', 'cj007_5_10_0', 'cj007_5_10_1',
        'cj007_2_3_0', 'cj009_2_3_1', 'cj009_5_10_0', 'cj009_5_10_1',
        'cj009_2_3_0', 'cj010_2_3_1', 'cj010_5_10_0', 'cj010_5_10_1',
        'cj010_2_3_0', 'cj011_2_3_1', 'cj011_5_10_0', 'cj011_5_10_1',
        'cj011_2_3_0', 'cj012_2_3_1', 'cj012_5_10_0', 'cj012_5_10_1',
        'cj012_2_3_0', 'cj013_2_3_1', 'cj013_5_10_0', 'cj013_5_10_1',
        'cj013_2_3_0', 'cj014_2_3_1', 'cj014_5_10_0', 'cj014_5_10_1',
        'cj014_2_3_0'
    ]
    other_columns = [
        f for f in factors_data.columns
        if f not in returns_columns + factors_columns + ['trade_time', 'code']
    ]

    target_dir = os.path.join(base_path, method, 'base', task_id)
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
    split_factors(method=variant.method, task_id=variant.task_id)
