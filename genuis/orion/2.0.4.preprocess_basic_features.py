### Crypto 进行数据预处理
import pdb, os, datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from lib.pre001.processing import factor_processing
from lib.pre001.winsorize import winsorize_normal
from lib.pre001.standardize import standardize
from kdutils.tactix import Tactix
from kdutils.macro2 import base_path


def preporcess(data, colmuns):
    data['trade_time'] = pd.to_datetime(data['trade_time'])
    data = data.set_index(['trade_time', 'code'])

    MIN_CROSS_SECTION_SIZE = 2
    # 计算每个时间点的样本数
    counts = data.groupby(level='trade_time').size()  # 如果是 MultiIndex
    # 或者 counts = train_data.groupby('trade_time').size() # 如果是列

    # 找到合规的时间点
    valid_times = counts[counts >= MIN_CROSS_SECTION_SIZE].index

    # 过滤数据
    data_clean = data[data.index.get_level_values('trade_time').isin(
        valid_times)]

    new_factors = factor_processing(
        data_clean[colmuns].values,
        pre_process=[winsorize_normal, standardize],
        groups=data_clean.index.get_level_values(0).values)
    factors_data = pd.DataFrame(new_factors,
                                columns=colmuns,
                                index=data_clean.index)
    return factors_data


## 加载已经处理切割数据 合并后进行标准化处理，再切割
def preprocess_basic_factors(method, period, source):
    base_dir1 = os.path.join(base_path, method, 'base', period, source)
    ## 加载基础特征数据
    train_data = pd.read_feather(os.path.join(base_dir1, "train_data.feather"))
    val_data = pd.read_feather(os.path.join(base_dir1, "val_data.feather"))
    test_data = pd.read_feather(os.path.join(base_dir1, "test_data.feather"))

    colmuns = [
        col for col in train_data.columns if col not in
        ['trade_time', 'code', 'f_funding_rate', 'f_funding_interval']
    ]

    train_data = preporcess(data=train_data, colmuns=colmuns)
    val_data = preporcess(data=val_data, colmuns=colmuns)
    test_data = preporcess(data=test_data, colmuns=colmuns)

    target_dir = os.path.join(base_path, method, 'normal', period, source)
    os.makedirs(target_dir, exist_ok=True)

    train_data.reset_index().to_feather(
        os.path.join(target_dir, "train_data.feather"))
    val_data.reset_index().to_feather(
        os.path.join(target_dir, "val_data.feather"))
    test_data.reset_index().to_feather(
        os.path.join(target_dir, "test_data.feather"))


def preprocess_rl_features(method, period, source):
    ## 读取因子
    pdb.set_trace()
    factor_dirs = os.path.join(base_path, method, 'normal', period, source)
    train_factors = pd.read_feather(
        os.path.join(factor_dirs, "train_data.feather"))
    train_factors['trade_time'] = pd.to_datetime(train_factors['trade_time'])
    val_factors = pd.read_feather(os.path.join(factor_dirs,
                                               "val_data.feather"))
    val_factors['trade_time'] = pd.to_datetime(val_factors['trade_time'])
    test_factors = pd.read_feather(
        os.path.join(factor_dirs, "test_data.feather"))
    test_factors['trade_time'] = pd.to_datetime(test_factors['trade_time'])

    ## 读取收益率
    return_dirs = os.path.join(base_path, method, 'base', period, source)
    train_return = pd.read_feather(
        os.path.join(return_dirs, "train_return.feather"))
    train_return['trade_time'] = pd.to_datetime(train_return['trade_time'])
    val_return = pd.read_feather(
        os.path.join(return_dirs, "val_return.feather"))
    val_return['trade_time'] = pd.to_datetime(val_return['trade_time'])
    test_return = pd.read_feather(
        os.path.join(return_dirs, "test_return.feather"))
    test_return['trade_time'] = pd.to_datetime(test_return['trade_time'])

    ## 合并数据
    total_factors = pd.concat([train_factors, val_factors, test_factors],
                              axis=0)

    total_return = pd.concat([train_return, val_return, test_return], axis=0)
    pdb.set_trace()
    ## 对齐数据
    total_data = total_factors.merge(total_return, on=['trade_time', 'code'])

    total_data = total_data.set_index(['trade_time', 'code']).unstack()
    total_data = total_data.fillna(0)  # 标准化填充 只能用0。在因子层面标识没有暴露，在收益率层面无收益
    total_data1 = total_data.stack(future_stack=True)

    ## 切割数据

    train_time = (train_factors['trade_time'].min(),
                  train_factors['trade_time'].max())
    val_time = (val_factors['trade_time'].min(),
                val_factors['trade_time'].max())
    test_time = (test_factors['trade_time'].min(),
                 test_factors['trade_time'].max())

    train_data = total_data1.loc[train_time[0]:train_time[1]]
    val_data = total_data1.loc[val_time[0]:val_time[1]]
    test_data = total_data1.loc[test_time[0]:test_time[1]]

    target_dir = os.path.join(base_path, method, 'rl', period, source)
    os.makedirs(target_dir, exist_ok=True)

    train_data.reset_index().to_feather(
        os.path.join(target_dir, "train_data.feather"))
    val_data.reset_index().to_feather(
        os.path.join(target_dir, "val_data.feather"))
    test_data.reset_index().to_feather(
        os.path.join(target_dir, "test_data.feather"))


if __name__ == '__main__':
    variant = Tactix().start()
    #preprocess_basic_factors(method=variant.method,
    #                         period=variant.period,
    #                         source=variant.source)

    preprocess_rl_features(method=variant.method,
                           period=variant.period,
                           source=variant.source)
