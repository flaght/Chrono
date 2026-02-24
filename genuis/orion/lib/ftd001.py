from kdutils.macro2 import *
from kdutils.logger import logger
import pandas as pd


### 读取数据 计算训练集，校验集，测试集，总数集的绩效
def fetch_temp_data(method, period, source, datasets, category='data'):

    res = []

    def fet(name, category):
        filename = os.path.join(base_path, method, 'base', period, source,
                                "{0}_{1}.feather".format(name, category))
        logger.info(filename)
        factors_data = pd.read_feather(filename).sort_values(
            by=['trade_time', 'code'])
        factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
        return factors_data

    for n in datasets:
        dt = fet(n, category)
        res.append(dt)

    res = pd.concat(res, axis=0)
    factors_data = res.sort_values(by=['trade_time', 'code'])
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    factors_data = factors_data.sort_values(by=['trade_time', 'code'])
    return factors_data


def fetch_temp_data1(method, source, task_id, datasets, category='data'):

    res = []

    def fet(name, category):
        filename = os.path.join(base_path, method, source, 'base', str(task_id),
                                "{0}_{1}.feather".format(name, category))
        logger.info(filename)
        factors_data = pd.read_feather(filename).sort_values(
            by=['trade_time', 'code'])
        factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
        return factors_data

    for n in datasets:
        dt = fet(n, category)
        res.append(dt)

    res = pd.concat(res, axis=0)
    factors_data = res.sort_values(by=['trade_time', 'code'])
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    factors_data = factors_data.sort_values(by=['trade_time', 'code'])
    return factors_data