import sys, os, torch, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from ultron.optimize.wisem.utilz.optimizer import to_device
from ultron.ump.similar.corrcoef import corr_xy, corr_matrix, ECoreCorrType

load_dotenv()
sys.path.append('../../kichaos')

from agent.envoy import Envoy0003


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
    return dt


class CogniDataSet2(Dataset):

    @classmethod
    def generate(cls, data, features, window, target=None, start_date=None):
        names = []
        res = []
        start_date = data.index.get_level_values(0).min().strftime(
            '%Y-%m-%d %H:%M:%S') if start_date is None else start_date
        #data_raw = data.set_index(['trade_time','code']).sort_index()
        data_raw = data.sort_index()
        raw = data_raw[features]
        uraw = raw.unstack()
        for i in range(0, window):
            names += ["{0}_{1}d".format(c, i) for c in features]
            res.append(uraw.shift(i).loc[start_date:].stack())

        dt = pd.concat(res, axis=1)
        dt.columns = names
        dt = dt.reindex(data_raw.index)
        dt = dt.loc[start_date:].dropna()
        if target is not None:
            dt = pd.concat([dt, data_raw[target]], axis=1)
            dt = dt.dropna().sort_index()
        return CogniDataSet2(dt, features, window=window, target=target)

    def __init__(self, data=None, features=None, window=None, target=None):
        if data is None:
            return
        self.data = data
        #names = list(set([f.split('_')[0] for f in features]))
        #self.features = names
        self.scaler = StandardScaler()
        self.features = features
        wfeatures = [f"{f}_{i}d" for f in features for i in range(window)]
        self.window = window
        self.wfeatures = wfeatures  # 时间滚动特征
        self.sfeatures = self.features  # 原始状态特征

        self.array = self.data[self.wfeatures].values
        self.array = torch.from_numpy(self.array).reshape(
            len(self.array), len(self.sfeatures), self.window)
        self.targets = None
        if isinstance(target, list):
            self.targets = self.data[target].values
            self.targets = torch.from_numpy(np.array(self.targets)).reshape(
                len(self.targets), 1)
        self.trade_time = self.data.index.get_level_values(0).strftime(
            '%Y-%m-%d %H:%M:%S')
        self.code = self.data.index.get_level_values(1)

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, index):
        date = self.trade_time[index]
        code = self.code[index]
        array = self.array[index]
        return {
            'trade_time': date,
            'code': code,
            'values': array
        } if self.targets is None else {
            'trade_time': date,
            'code': code,
            'values': array,
            'target': self.targets[index]
        }


def fetch_data(code, method, scaler_name, horizon, offset=1):
    file_name = os.path.join(
        os.getenv('BASE_PATH'), 'times', code,
        f'{method}_lumina_{scaler_name}_{horizon}_{offset}_factors.feather')
    data = pd.read_feather(file_name)
    return data.dropna(subset=['nxt1_ret'])


def load_data(method):
    file_path = os.path.join(os.getenv('BASE_PATH'), 'times', 'factors',
                             f'{method}_lumina_features.feather')
    data = pd.read_feather(file_path)
    return data


def load_data1(method, horizon):
    #data = fetch_data('factors', method, os.getenv('SCALER'), int(horizon))
    #data = fetch_data('IH', method, os.getenv('SCALER'), int(horizon))
    file_name = os.path.join(
        os.getenv('BASE_PATH'), 'times', 'factors',
        f'{method}_{horizon}_factors.feather')
    data = pd.read_feather(file_name)
    return data.dropna(subset=['nxt1_ret'])

    return data


def create_data(codes, window=3):
    
    val_data = load_data1('min_val', os.getenv('HORIZON')).sort_values(
        ['trade_time', 'code']).set_index(['trade_time', 'code'])
    train_data = load_data1('min_train', os.getenv('HORIZON')).sort_values(
        ['trade_time', 'code']).set_index(['trade_time', 'code'])

    codes = train_data.index.get_level_values(1).unique().tolist()
    features = [
        col for col in train_data.columns
        if col not in ['chg', 'price', 'nxt1_ret']
    ]
    #features = [
    #    'bias_2', 'bias_3', 'pgo_2', 'pgo_3', 'psl_3', 'willr_3', 'rsi_2',
    #    'willr_2', 'pvol_1', 'bop_1'
    #]
    #features = features[0:5]
    #train_data[features] = train_data[features] * -1
    #val_data[features] = val_data[features] * -1

    #train_data['nxt1_ret_1s'] = train_data['nxt1_ret']
    #train_data['nxt1_ret_2s'] = train_data['nxt1_ret']
    #val_data['nxt1_ret_1s'] = val_data['nxt1_ret']
    #val_data['nxt1_ret_2s'] = val_data['nxt1_ret']
    #
    #features = ['nxt1_ret_1s', 'nxt1_ret_2s']

    train_data = train_data[features + ['nxt1_ret']]
    val_data = val_data[features + ['nxt1_ret']]

    train_sets = CogniDataSet2.generate(train_data,
                                        features=features,
                                        window=window,
                                        target=['nxt1_ret'])

    val_sets = CogniDataSet2.generate(val_data,
                                      features=features,
                                      window=window,
                                      target=['nxt1_ret'])
    train_loader = DataLoader(dataset=train_sets,
                              batch_size=512,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_sets, batch_size=512, shuffle=False)

    return train_loader, val_loader, features, codes


def train(window, codes):
    train_loader, val_loader, features, codes = create_data(codes, window)
    envoy = Envoy0003(id="{0}s_{1}h_{2}w".format(os.environ['SCALER'],
                                                 os.getenv('HORIZON'), window),
                      features=features,
                      ticker_count=len(codes),
                      window=window)
    max_episode = 50
    envoy._temporal_hybrid_transformer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        is_state_dict=True,
        model_dir=envoy.train_path,
        tb_dir=envoy.tensorboard_path,
        push_dir=envoy.push_path,
        epochs=max_episode)

    #envoy.train_model(train_loader=train_loader,
    #                  val_loader=val_loader,
    #                  is_state_dict=True,
    #                  epochs=max_episode)


def predict(window, codes):
    
    data = load_data(os.getenv('METHOD'))
    nxt1_ret = create_yields(data, int(os.getenv('HORIZON')))

    train_loader, val_loader, features, codes = create_data(codes, window)
    envoy = Envoy0003(id="{0}s_{1}h_{2}w".format(os.environ['SCALER'],
                                                 os.getenv('HORIZON'), window),
                      features=features,
                      ticker_count=len(codes),
                      window=window,
                      is_load=True)
    res = []

    for data in val_loader:
        X = to_device(data['values'])
        y = to_device(data['target'])
        outputs = envoy._temporal_hybrid_transformer.predict(X, True)

        data = {
            'trade_time': data['trade_time'],
            'code': data['code'],
            'value': outputs.detach().cpu().numpy().reshape(-1),
            #'nxt1_ret': y.detach().cpu().numpy().reshape(-1)
        }
        data = pd.DataFrame(data).set_index(['trade_time', 'code'])
        res.append(data)
    
    factors_data = pd.concat(res, axis=0)
    factors_data = factors_data.reset_index()
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    factors_data = factors_data.merge(nxt1_ret, on=['trade_time','code'])
    '''
    factors_data.index.set_levels(pd.to_datetime(factors_data.index.levels[0]),
                                  level=0,
                                  inplace=True)
    '''
    
    corr_value = corr_xy(factors_data['value'], factors_data['nxt1_ret'],
                 ECoreCorrType.E_CORE_TYPE_SPERM)
    print(f'Corr: {corr_value}')


def main():
    codes = ['IH', 'IC', 'IF', 'IM']
    #train(window=3, codes=codes)
    predict(window=3,codes=codes)


main()
