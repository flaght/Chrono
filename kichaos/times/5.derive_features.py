import os, pdb
import pandas as pd

from dotenv import load_dotenv

pd.options.mode.copy_on_write = True
load_dotenv()

from ultron.factor.genetic.geneticist.operators import calc_factor
from ultron.ump.similar.corrcoef import ECoreCorrType
from ultron.ump.similar.corrcoef import corr_xy, ECoreCorrType
from ultron.tradingday import advanceDateByCalendar
from sklearn.preprocessing import minmax_scale

def fetch_evolution():
    file_name = os.path.join(
        os.getenv('BASE_PATH'), 'times', 'evolition',
        "evolition_{}_{}.feather".format(os.environ['HORIZON'], '108990000'))
    data = pd.read_feather(file_name)
    data = data[data['fitness'] > 0.03][['name', 'formual', 'fitness']]
    #print(data[['formual','fitness']].head(10))
    return data.values.tolist()


## 收益率处理
def create_yields(data, horizon, offset=1):
    dt = data.set_index(['trade_time'])
    dt["nxt1_ret"] = dt['chg']
    dt = dt.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    dt = dt.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    dt.name = 'nxt1_ret'
    data = dt.reset_index().merge(data, on=['trade_time',
                                            'code']).reset_index(drop=True)
    return data.dropna(subset=['nxt1_ret'])


def load_data(method, horizon):
    file_path = os.path.join(os.getenv('BASE_PATH'), 'times', 'factors',
                             f'{method}_lumina_features.feather')
    data = pd.read_feather(file_path)
    return data


## 计算挖掘因子
def create_features():
    ## 读取进化表达式
    express = fetch_evolution()
    ## 读取基础因子
    
    data = load_data(os.getenv('METHOD'), os.getenv('HORIZON'))
    data = data.sort_values(by=['trade_time', 'code'])
    #nxt1_ret = data[['trade_time', 'code', 'nxt1_ret']]
    res = []
    result = []
    for exp in express[0:30]:
        print(exp)
        name = str(exp[0])
        factor_data = calc_factor(expression=exp[1],
                                  total_data=data,
                                  name=name,
                                  indexs=['trade_time'],
                                  key='code')
        factor_data = factor_data.reset_index().sort_values(
            by=['trade_time', 'code'])

        #corr_value = corr_xy(factor_data[name], nxt1_ret['nxt1_ret'],
        #                     ECoreCorrType.E_CORE_TYPE_SPERM)
        res.append(factor_data.set_index(['trade_time', 'code']))
        #result.append({
        #    'name': exp[0],
        #    'e_corr': exp[-1],
        #    'n_corr': corr_value
        #})
    filename = os.path.join(
        os.getenv('BASE_PATH'), 'times', 'factors',
        f'{os.getenv("METHOD")}_ultron_features.feather')
    factors_data = pd.concat(res, axis=1)
    factors_data.reset_index().to_feather(filename)
    
    print(result)


def process_features():
    method = os.getenv('METHOD')
    file_path = os.path.join(os.getenv('BASE_PATH'), 'times', 'factors',
                             f'{method}_lumina_features.feather')
    lumina_data = pd.read_feather(file_path)

    file_path = os.path.join(os.getenv('BASE_PATH'), 'times', 'factors',
                             f'{method}_ultron_features.feather')
    ultron_data = pd.read_feather(file_path)
    ## 过滤掉 有nan值因子
    filter_list = ['ultron_1720488161727862','ultron_1720487354911916','ultron_1720494218729302']
    ultron_data = ultron_data.drop(columns=filter_list)
    factors_data = lumina_data.merge(ultron_data, on=['trade_time', 'code'])

    start_date = advanceDateByCalendar('china.sse', factors_data['trade_time'].min(),
                                       '2b')
    factors_data = factors_data[factors_data.trade_time >= start_date]

    features = [
        col for col in factors_data.columns
        if col not in ['trade_time', 'code', 'price', 'chg', 'close', 'high', 'low', 'open', 'value', 'volume', 'openint']
    ]

    factors_data[features] = minmax_scale(factors_data[features].values, feature_range=(-1, 1))
    return factors_data[['trade_time','code'] + features + ['chg']]

def create_yields(data, horizon, offset=1):
    
    method = os.getenv('METHOD')
    dt = data.set_index(['trade_time'])
    dt["nxt1_ret"] = dt['chg']
    dt = dt.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    dt = dt.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    dt.name = 'nxt1_ret'
    dt = dt.reset_index()
    dt[['nxt1_ret']] = minmax_scale(dt[['nxt1_ret']].values,
                                    feature_range=(-1, 1))
    data = data.merge(dt, on=['trade_time',
                              'code']).sort_values(by=['trade_time', 'code'])
    data = data.dropna(subset=['nxt1_ret'])

    file_name = os.path.join(
        os.getenv('BASE_PATH'), 'times', 'factors',
        f'{method}_{horizon}_factors.feather')
    print(f'Saving {file_name}')
    data.reset_index(drop=True).to_feather(file_name)

    
def process_data():
    factors_data = process_features()
    for horizon in [1, 2, 5]:
        create_yields(factors_data.copy(), horizon)

create_features()
process_data()
