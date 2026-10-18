## 加载外部tick降频1分钟特征，进行切割
import os, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from kdutils.data import filter_trading_time
from kdutils.macro import base_path
from kdutils.macro2 import TRADING_TIME_MAPPING
from kdutils.ttimes import get_dates, FIIXED_MAPPING
from kdutils.tactix import Tactix


def run1(method, instruments, split):
    base_dirs = os.path.join(os.environ['SIRIUS_DIRS'], method, instruments,
                             'factors')
    res = []
    for root, dirs, files in os.walk(base_dirs):
        for file in files:
            print(file)
            if file.endswith('.feather') and file != 'factors_data.feather':
                factor_file = os.path.join(root, file)
                factor_data = pd.read_feather(factor_file)
                res.append(
                    factor_data.set_index(['trade_time', 'code', 'symbol']))
    factors_data = pd.concat(res, axis=1).sort_index()
    total_rows = len(factors_data)
    coverage = factors_data.count() / total_rows
    low_coverage = coverage[coverage < 0.9]
    low_columns = low_coverage.index.tolist()
    factors_data = factors_data.drop(low_columns, axis=1)
    ## 前值填充
    pdb.set_trace()
    # factors_data = factors_data.unstack().fillna(method='ffill')
    factors_data = factors_data.groupby(level='code').ffill()
    # factors_data = factors_data.stack()
    nan_columns = factors_data.columns[factors_data.isna().all()]
    factors_data = factors_data.drop(nan_columns, axis=1)
    factors_data = factors_data.dropna().reset_index()
    start_date = factors_data['trade_time'].min().strftime('%Y-%m-%d %H:%M:%S')
    end_date = factors_data['trade_time'].max().strftime('%Y-%m-%d %H:%M:%S')

    factors_data['trade_time'] = pd.to_datetime(
        factors_data['trade_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    factors_data = factors_data.sort_values(by=['trade_time', 'code'])

    factors_data = filter_trading_time(
        factors_data,
        trading_sessions=TRADING_TIME_MAPPING['RB'],
        is_reset_index=False,
        time_name='trade_time')
    ## 非正常周期过滤
    # factors_data = filter_date(total_data=factors_data,
    #                            start_exclude='2015-07-07 14:35:00',
    #                            end_exclude='2015-08-25 09:50:00')
    if split == 'scale':
        times = factors_data['trade_time'].unique().tolist()
        len1 = round(len(times) * 0.6)  # 60%部分
        len2 = round(len(times) * 0.25)  # 25%部分
        len3 = len(times) - len1 - len2

        ## 训练集

        train_data = factors_data[factors_data['trade_time'].isin(
            times[:len1])]
        val_data = factors_data[factors_data['trade_time'].isin(
            times[len1:len1 + len2])]
        test_data = factors_data[factors_data['trade_time'].isin(times[len1 +
                                                                       len2:])]
    elif split == 'fixed':
        train_data = factors_data[factors_data['trade_time'] <=
                                  FIIXED_MAPPING[method]['train_end']]
        val_data = factors_data[
            (factors_data['trade_time'] > FIIXED_MAPPING[method]['train_end'])
            &
            (factors_data['trade_time'] <= FIIXED_MAPPING[method]['val_end'])]
        test_data = factors_data[(factors_data['trade_time']
                                  > FIIXED_MAPPING[method]['val_end'])]
    ## 校验集
    ## 测试集
    ### 切割数据
    pdb.set_trace()
    target_dir = os.path.join(base_path, method, instruments, 'micro')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    train_data.reset_index(drop=True).to_feather(
        os.path.join(base_path, method, instruments, 'micro',
                     'train_data.feather'))
    val_data.reset_index(drop=True).to_feather(
        os.path.join(base_path, method, instruments, 'micro',
                     'val_data.feather'))
    test_data.reset_index(drop=True).to_feather(
        os.path.join(base_path, method, instruments, 'micro',
                     'test_data.feather'))


if __name__ == '__main__':
    #variant = Tactix().start()
    run1(method='ricso2', instruments='hcb', split='fixed')
