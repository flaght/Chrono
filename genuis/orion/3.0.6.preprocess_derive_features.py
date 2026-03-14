import pdb, os, datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from lib.pre001.processing import factor_processing
from lib.pre001.winsorize import winsorize_normal
from lib.pre001.standardize import standardize
from kdutils.tactix import Tactix
from kdutils.macro2 import base_path
from kdutils.logger import logger


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
    logger.info(f"process data {data_clean.shape}")
    new_factors = factor_processing(
        data_clean[colmuns].values,
        pre_process=[winsorize_normal, standardize],
        groups=data_clean.index.get_level_values(0).values)
    factors_data = pd.DataFrame(new_factors,
                                columns=colmuns,
                                index=data_clean.index)
    return factors_data


## 加载已经处理切割数据 合并后进行标准化处理，再切割
def preprocess_derive_factors(method, task_id, session):
    factor_path = os.path.join(base_path, method, 'evaluate', str(task_id),
                               'results', session, "data.feather")
    logger.info(f"load data {factor_path}")

    total_factors = pd.read_feather(factor_path)
    pdb.set_trace()
    ## 前置填充
    colmuns = [
        col for col in total_factors.columns
        if col not in ['trade_time', 'code']
    ]
    total_factors[colmuns] = total_factors[colmuns].replace([np.inf, -np.inf], np.nan)
    total_factors = total_factors.dropna()
    total_factors['trade_time'] = pd.to_datetime(total_factors['trade_time'])
    ## 去极值标准化
    total_factors = preporcess(data=total_factors, colmuns=colmuns)
    ## 读取收益率
    return_dirs = os.path.join(base_path, method, 'base', task_id)
    train_return = pd.read_feather(
        os.path.join(return_dirs, "train_return.feather"))
    train_return['trade_time'] = pd.to_datetime(train_return['trade_time'])
    val_return = pd.read_feather(
        os.path.join(return_dirs, "val_return.feather"))
    val_return['trade_time'] = pd.to_datetime(val_return['trade_time'])
    test_return = pd.read_feather(
        os.path.join(return_dirs, "test_return.feather"))
    test_return['trade_time'] = pd.to_datetime(test_return['trade_time'])

    # 合并收益率
    total_return = pd.concat([train_return, val_return, test_return], axis=0)
    total_data = total_factors.merge(total_return, on=['trade_time', 'code'])
    total_data['trade_time'] = pd.to_datetime(total_data['trade_time'])
    # total_data = total_data.set_index(['trade_time', 'code']).unstack()
    total_data = total_data.set_index(['trade_time', 'code'])
    total_data1 = total_data.fillna(0)  # 标准化填充 只能用0。在因子层面标识没有暴露，在收益率层面无收益
    # total_data1 = total_data.stack()

    ## 切割数据
    pdb.set_trace()
    train_time = (train_return['trade_time'].min(),
                  train_return['trade_time'].max())
    val_time = (val_return['trade_time'].min(), val_return['trade_time'].max())
    test_time = (test_return['trade_time'].min(),
                 test_return['trade_time'].max())

    train_data = total_data1.loc[train_time[0]:train_time[1]]
    val_data = total_data1.loc[val_time[0]:val_time[1]]
    test_data = total_data1.loc[test_time[0]:test_time[1]]

    target_dir = os.path.join(base_path, method, 'rl', task_id)
    os.makedirs(target_dir, exist_ok=True)

    train_data.reset_index().to_feather(
        os.path.join(target_dir, "derive_train_data.feather"))
    val_data.reset_index().to_feather(
        os.path.join(target_dir, "derive_val_data.feather"))
    test_data.reset_index().to_feather(
        os.path.join(target_dir, "derive_test_data.feather"))


if __name__ == '__main__':
    variant = Tactix().start()
    preprocess_derive_factors(method=variant.method,
                              task_id=variant.task_id,
                              session=variant.session)
