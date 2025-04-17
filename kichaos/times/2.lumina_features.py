import os, pdb, inspect, re, time
import pandas as pd
import numpy as np
from dotenv import load_dotenv

pd.options.mode.copy_on_write = True
load_dotenv()

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import minmax_scale
import lumina.techinical as lt
import lumina.features as lf
from ultron.tradingday import advanceDateByCalendar
from ultron.strategy.models.processing import standardize as alk_standardize
from ultron.strategy.models.processing import winsorize as alk_winsorize

import numpy as np


class CustomStandardScaler:

    def __init__(self):
        self.scale_ = None

    def fit(self, X):
        """
        计算每个特征的标准差.
        
        参数:
        X (ndarray): 形状为 (n_samples, n_features) 的训练数据.
        
        返回:
        self: 拟合好的 scaler.
        """
        X = np.asarray(X)
        self.scale_ = np.std(X, axis=0, ddof=0)
        return self

    def transform(self, X):
        """
        使用标准差缩放数据.
        
        参数:
        X (ndarray): 形状为 (n_samples, n_features) 的数据.
        
        返回:
        X_scaled (ndarray): 缩放后的数据.
        """
        if self.scale_ is None:
            raise RuntimeError(
                "This CustomStandardScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        X = np.asarray(X)
        return X / self.scale_

    def fit_transform(self, X):
        """
        拟合数据并进行标准差缩放.
        
        参数:
        X (ndarray): 形状为 (n_samples, n_features) 的数据.
        
        返回:
        X_scaled (ndarray): 缩放后的数据.
        """
        return self.fit(X).transform(X)


def convert_feature_name(name):
    # 将字符串中的 Feature 替换为空字符串
    converted_name = re.sub(r'^Feature', '', name)

    # 将连续的大写字母转换为小写字母，并在其间添加下划线
    converted_name = re.sub(r'([a-z])([A-Z])', r'\1_\2',
                            converted_name).lower()

    return converted_name


def load_data(method, code):
    file_path = os.path.join(os.getenv('BASE_PATH'), 'times', code,
                             f'{method}.feather')
    print(f'Loading {file_path}')
    data = pd.read_feather(file_path)
    return data.reset_index(drop=True)


def fetch_features():
    # FeatureExtMean FeatureExtVar 长度不一样，
    t1 = [
        'FeatureBase', 'FeatureAtr', 'FeatureExtVar', 'FeatureAnnealn',
        'FeatureMaximumSum', 'FeatureMaximumMean', 'FeatureMinimumMean',
        'FeatureMinimumSum', 'FeatureThick'
    ]
    features_list = [
        func for func, obj in inspect.getmembers(lf) if inspect.isclass(obj)
    ]
    features_list = [
        func for func in features_list if func.startswith('Feature')
    ]
    features_list = [func for func in features_list if func not in t1]
    return features_list
    #getattr(lf,'FeatureAdosc')


def calc_features(data, features_list):
    start_time = time.time()
    res = []
    indexs = data.set_index(['trade_time', 'code']).index
    count, _ = data.shape
    for feature in features_list:
        print(f'Calculating {feature}')
        #feature = 'FeaturePVOL'
        keys_vars = {
            attr: value
            for attr, value in getattr(lf, feature)().__dict__.items()
            if attr.endswith('_keys')
        }
        key_name = list(keys_vars.keys())[0]
        keys = keys_vars[key_name]
        name = key_name.split('_keys')[0]
        i = 0
        for k in keys:
            i += 1
            params = k if isinstance(k, tuple) else (k, )
            ny = getattr(lt, "calc_{0}".format(name.lower()))(data, *params)
            if isinstance(ny, tuple):
                for j in range(len(ny)):
                    fname = "{0}_{1}_{2}".format(name, i, j)
                    dt = pd.Series(ny[j].tl, index=indexs, name=fname)
                    assert (dt.dropna().shape[0] == count)
                    res.append(dt)
            else:
                fname = "{0}_{1}".format(name, i)
                dt = pd.Series(ny.tl, index=indexs, name=fname)
                assert (dt.dropna().shape[0] == count)
                res.append(dt)
    end_time = time.time()
    print(f'Time: {end_time - start_time}')
    factors_data = pd.concat(res, axis=1).reset_index()
    factors_data = factors_data.merge(data, on=['trade_time', 'code'])
    return factors_data.sort_values(['trade_time', 'code'])


def update_features(codes, method):
    res = []
    for code in codes:
        features_list = fetch_features()
        data = load_data(method=method, code=code).dropna(subset=['chg'])
        factors = calc_features(data, features_list)
        res.append(factors)

    #features = [
    #        col for col in factors.columns if col not in ['trade_time', 'code']
    #]
    
    factors = pd.concat(res, axis=0).sort_values(['trade_time', 'code'])
    file_path = os.path.join(os.getenv('BASE_PATH'), 'times', "factors")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    filename = os.path.join(file_path, f'{method}_lumina_features.feather')
    print(f'Saving {filename}')
    factors.reset_index(drop=True).to_feather(filename)


## 标准化处理
def process_features(method, scaler_name):
    if scaler_name == 'standard':
        scaler = StandardScaler()
    elif scaler_name == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_name == 'custom':
        scaler = CustomStandardScaler()
    elif scaler_name == 'ultron':
        scaler = StandardScaler()
    else:
        raise ValueError('Invalid scaler name')

    file_path = os.path.join(os.getenv('BASE_PATH'), 'times', 'factors',
                             f'{method}_lumina_features.feather')
    print(f'Loading {file_path}')

    data = pd.read_feather(file_path)

    start_date = advanceDateByCalendar('china.sse', data['trade_time'].min(),
                                       '2b')
    data = data[data.trade_time >= start_date]
    features = [
        col for col in data.columns
        if col not in ['trade_time', 'code', 'price', 'chg']
    ]
    #data[features] = data[features].replace([-np.inf, np.inf],np.nan)
    data[features] = minmax_scale(data[features].values, feature_range=(-1, 1))
    #data[features] = scaler.fit_transform(data[features].values)
    assert (data.dropna().shape[0] == data.shape[0])
    file_name = os.path.join(
        os.getenv('BASE_PATH'), 'times', 'factors',
        f'{method}_lumina_{scaler_name}_features.feather')
    print(f'Saving {file_name}')
    data.reset_index(drop=True).to_feather(file_name)


## 收益率处理
def create_yields(method, scaler_name, horizon, offset=1):
    file_path = os.path.join(
        os.getenv('BASE_PATH'), 'times', 'factors',
        f'{method}_lumina_{scaler_name}_features.feather')
    print(f'Loading {file_path}')
    data = pd.read_feather(file_path)

    dt = data.set_index(['trade_time'])
    dt["nxt1_ret"] = dt['chg']
    dt = dt.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    dt = dt.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    dt.name = 'nxt1_ret'
    dt.reset_index().merge(data, on=['trade_time',
                                     'code']).reset_index(drop=True)
    '''
    if scaler_name in ['standard', 'minmax', 'custom']:
        if scaler_name == 'standard':
            scaler = StandardScaler()
        elif scaler_name == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_name == 'custom':
            scaler = CustomStandardScaler()
        ##缩放 -1~1
        dt.name = 'nxt1_ret'
        dt = dt.reset_index()
        dt['nxt1_ret'] = scaler.fit_transform(dt[['nxt1_ret']].values)
        min_val = np.min(dt['nxt1_ret'])
        max_val = np.max(dt['nxt1_ret'])
        data_scaled = 2 * (dt['nxt1_ret']  - min_val) / (max_val - min_val) - 1
        dt['nxt1_ret'] = data_scaled

    elif scaler_name == 'ultron':
        dt = alk_standardize(alk_winsorize(dt.dropna().unstack())).unstack()
        dt.name = 'nxt1_ret'
    '''
    dt.name = 'nxt1_ret'
    dt = dt.reset_index()
    dt[['nxt1_ret']] = minmax_scale(dt[['nxt1_ret']].values,
                                    feature_range=(-1, 1))
    data = data.merge(dt, on=['trade_time',
                              'code']).sort_values(by=['trade_time', 'code'])
    data = data.dropna(subset=['nxt1_ret'])
    file_name = os.path.join(
        os.getenv('BASE_PATH'), 'times', 'factors',
        f'{method}_lumina_{scaler_name}_{horizon}_{offset}_factors.feather')
    print(f'Saving {file_name}')
    data.reset_index(drop=True).to_feather(file_name)


def main(method):
    codes = ['IM', 'IH', 'IC', 'IF']
    update_features(codes=codes, method=method)
    process_features(method, os.getenv('SCALER'))
    for horizon in [1, 2, 5]:
        create_yields(method, os.getenv('SCALER'), int(horizon))


main(os.getenv('METHOD'))
