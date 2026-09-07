### 做挖掘因子计算 调整方向， 时序标准化
import os
import pandas as pd
from dotenv import load_dotenv
import pdb

load_dotenv()

from kdutils.macro2 import base_path
from kdutils.tactix import Tactix
from lib.aux001 import fetch_temp_returns
from lib.syn001 import build_factors
from lib.composite.loader import DataLoader


## 生成因子包括调整方向，时序标准化
def create_normal_factors1(method, instruments, task_id, period, name):
    build_factors(method=method,
                  instruments=instruments,
                  task_id=task_id,
                  period=period,
                  name=name)


## 切割因子数据，创建训练集 校验集 测试集
def prepare(method, instruments, task_id, period, name):
    pdb.set_trace()
    train_data, val_data, test_data = DataLoader().load_from_project(
        method=method,
        task_id=task_id,
        instruments=instruments,
        period=period,
        name="final_{0}".format(name),
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

    train_data = train_data.merge(
        train_return[['trade_time', 'code', 'nxt1_ret_1h']],
        on=['trade_time', 'code'])

    val_data = val_data.merge(val_return[['trade_time', 'code',
                                          'nxt1_ret_1h']],
                              on=['trade_time', 'code'])

    test_data = test_data.merge(
        test_return[['trade_time', 'code', 'nxt1_ret_1h']],
        on=['trade_time', 'code'])

    train_data['trade_time'] = pd.to_datetime(train_data['trade_time'])
    val_data['trade_time'] = pd.to_datetime(val_data['trade_time'])
    test_data['trade_time'] = pd.to_datetime(test_data['trade_time'])

    output_dirs = os.path.join(base_path, method, instruments, 'temp', 'model',
                               str(task_id), str(period), 'rl', 'data')
    os.makedirs(output_dirs, exist_ok=True)

    train_data.reset_index(drop=True).to_feather(
        os.path.join(output_dirs, "train_data.feather"))
    val_data.reset_index(drop=True).to_feather(
        os.path.join(output_dirs, "val_data.feather"))
    test_data.reset_index(drop=True).to_feather(
        os.path.join(output_dirs, "test_data.feather"))


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == 'build':
        create_normal_factors1(method=variant.method,
                               instruments=variant.instruments,
                               task_id=variant.task_id,
                               period=variant.period,
                               name=variant.name)
    elif variant.form == 'prepare':
        prepare(method=variant.method,
                instruments=variant.instruments,
                task_id=variant.task_id,
                period=variant.period,
                name=variant.name)
