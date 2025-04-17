### 混合模型,
### 长期模型特征 + 短期模型特征 + 多模态特征 == 特征

import sys, os, torch, pdb, argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from ultron.optimize.wisem import *
from ultron.optimize.wisem.utilz.optimizer import to_device
from ultron.strategy.models.processing import standardize as alk_standardize
from ultron.strategy.models.processing import winsorize as alk_winsorize

load_dotenv()
sys.path.append('../../kichaos')

from datasets import CogniDataSet3
from datasets import CogniDataSet6
from datasets import CogniDataSet8
from agent import Envoy0002
from agent import Liaison0004
from agent import Pawn0003
from utils.env import *

from kdutils.macro import base_path
from kdutils.metrics import metrics as kdmetrics
from kdutils.data import *

### 使用模型预测数据，作为混合模型的准备, 使用预测方法


### 时序模型预测，中间层输出
def load_timing_misro(method,
                      window=3,
                      seq_cycle=10,
                      horizon=1,
                      universe=None,
                      time_format='%Y-%m-%d'):
    filename = os.path.join(base_path, universe,
                            "{0}_model_normal.feather".format(method))
    total_data = pd.read_feather(filename)
    total_data = total_data.sort_values(['trade_date', 'code'])
    nxt1_columns = total_data.filter(regex="^nxt1_").columns.to_list()

    columns = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code'] + nxt1_columns
    ]

    total_data = total_data[['trade_date', 'code'] + columns +
                            ['nxt1_ret_{0}h'.format(horizon)]].sort_values(
                                ['trade_date', 'code'])

    total_data.rename(columns={'nxt1_ret_{0}h'.format(horizon): 'nxt1_ret'},
                      inplace=True)
    total_data = total_data.sort_values(['trade_date', 'code'])

    dates = total_data['trade_date'].dt.strftime(time_format).unique().tolist()
    #pos = int(len(dates) * 0.7)
    pos = 0 + window + seq_cycle + 1 + 1
    factors_data = total_data[total_data['trade_date'].isin(dates[pos:])]
    features = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code', 'dummy', 'nxt1_ret']
    ]
    window_val_data = CogniDataSet3.create_windows(data=factors_data.set_index(
        ['trade_date', 'code']),
                                                   features=features,
                                                   window=window,
                                                   time_name='trade_date')
    return window_val_data


### MAE模型预测，中间层输出
def load_mae_misro(method, window, universe, time_format='%Y-%m-%d'):
    filename = os.path.join(base_path, universe,
                            "{0}_model_normal.feather".format(method))
    total_data = pd.read_feather(filename)
    total_data = total_data.sort_values(['trade_date', 'code'])

    nxt1_columns = total_data.filter(regex="^nxt1_").columns.to_list()
    columns = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code'] + nxt1_columns
    ]
    #columns = columns[0:15]
    total_data = total_data[['trade_date', 'code'] + columns +
                            nxt1_columns].sort_values(['trade_date', 'code'])
    dates = total_data['trade_date'].dt.strftime(time_format).unique().tolist()

    pos = 0 + window + 1

    factors_data = total_data[total_data['trade_date'].isin(dates[pos:])]

    features = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code', 'dummy', 'nxt1_ret'] +
        nxt1_columns
    ]
    pdb.set_trace()
    window_val_data = CogniDataSet6.create_windows(data=factors_data.set_index(
        ['trade_date', 'code']),
                                                   features=features,
                                                   target=nxt1_columns,
                                                   window=window,
                                                   time_name='trade_date')
    return window_val_data


### 加载预测数据 生成特征数据 预测使用
def load_misro(method, window, universe, horizon=1, time_format='%Y-%m-%d'):
    total_data = pd.read_feather(
        os.path.join(base_path, universe,
                     "{0}_model_factors_normal.feather".format(method)))
    long_temporal_features = total_data.filter(
        regex="^factor_long").columns.to_list()
    short_temporal_features = total_data.filter(
        regex="^factor_short").columns.to_list()
    mae_features = total_data.filter(regex="^hidden").columns.to_list()
    features = mae_features + long_temporal_features + short_temporal_features

    total_data = total_data.sort_values(['trade_date', 'code'])
    dates = total_data['trade_date'].dt.strftime(time_format).unique().tolist()

    pdb.set_trace()
    pos = int(len(dates) * 0.7)

    _ = total_data[total_data['trade_date'].isin(dates[:pos])]
    val_data = total_data[total_data['trade_date'].isin(dates[pos - window +
                                                              1:])]
    window_val_data = CogniDataSet8.create_windows(data=val_data.set_index(
        ['trade_date', 'code']),
                                                   features=features,
                                                   window=window,
                                                   time_name='trade_date')
    return window_val_data, mae_features, long_temporal_features, short_temporal_features


### 加载预测数据，生成特征数据 训练使用
def load_micro(method,
               window,
               seq_cycle,
               universe,
               horizon=1,
               time_format='%Y-%m-%d'):
    '''
    ## 加载 时序数据
    long_filename = os.path.join(
        base_path, universe,
        "{0}_long_envoy0002_{1}c_{2}w.feather".format(method, seq_cycle, 3))

    short_filename = os.path.join(
        base_path, universe,
        "{0}_short_envoy0002_{1}c_{2}w.feather".format(method, seq_cycle, 3))

    long_temporal_data = pd.read_feather(long_filename)
    short_temporal_data = pd.read_feather(short_filename)

    ## 加载 MAE数据
    mae_filename = os.path.join(
        base_path, universe, "{0}_liaison0004_hidden.feather".format(method))
    mae_temporal_data = pd.read_feather(mae_filename)
    


    ### MAE 数据  long 数据， short 数据
    total_data = mae_temporal_data.merge(long_temporal_data,
                                         on=['trade_date', 'code']).merge(
                                             short_temporal_data,
                                             on=['trade_date', 'code'])
    '''
    total_data = pd.read_feather(
        os.path.join(base_path, universe,
                     "{0}_model_factors_normal.feather".format(method)))

    long_temporal_features = total_data.filter(
        regex="^factor_long").columns.to_list()
    short_temporal_features = total_data.filter(
        regex="^factor_short").columns.to_list()
    mae_features = total_data.filter(regex="^hidden").columns.to_list()

    ## 加载收益率
    filename = os.path.join(base_path, universe,
                            "{0}_model_normal.feather".format(method))
    yield_data = pd.read_feather(filename)
    yield_data = yield_data.sort_values(['trade_date', 'code'])
    yield_data = yield_data[['trade_date', 'code'] +
                            ['nxt1_ret_{0}h'.format(horizon)]].sort_values(
                                ['trade_date', 'code'])
    total_data = total_data.merge(yield_data, on=['trade_date', 'code'])
    total_data.rename(columns={'nxt1_ret_{0}h'.format(horizon): 'nxt1_ret'},
                      inplace=True)
    features = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code', 'dummy', 'nxt1_ret']
    ]
    total_data = total_data.sort_values(['trade_date', 'code'])
    dates = total_data['trade_date'].dt.strftime(time_format).unique().tolist()
    pos = int(len(dates) * 0.7)

    train_data = total_data[total_data['trade_date'].isin(dates[:pos])]
    val_data = total_data[total_data['trade_date'].isin(dates[pos:])]

    codes = total_data.code.unique().tolist()

    train_dataset = CogniDataSet8.generate(train_data,
                                           codes=codes,
                                           features=features,
                                           window=window,
                                           target=['nxt1_ret'],
                                           time_name='trade_date',
                                           time_format=time_format)

    val_dataset = CogniDataSet8.generate(val_data,
                                         codes=codes,
                                         features=features,
                                         window=window,
                                         target=['nxt1_ret'],
                                         time_name='trade_date',
                                         time_format=time_format)
    return train_dataset, val_dataset, mae_features, long_temporal_features, short_temporal_features


def create_timing_data(variant):
    window_data = load_timing_misro(method=variant['method'],
                                    window=variant['window'],
                                    seq_cycle=variant['seq_cycle'],
                                    horizon=variant['horizon'],
                                    universe=variant['universe'])

    envoy = Envoy0002(id="dubhe_{0}_{1}c_{2}".format(variant['method'],
                                                     variant['seq_cycle'],
                                                     variant['universe']),
                      features=window_data['features'],
                      ticker_count=len(window_data['code'].unique()),
                      window=variant['window'],
                      is_load=True,
                      is_debug=False)

    hidden = envoy.dimension

    factors_long_name = [
        "factor_long_{0}".format(i) for i in range(0, int(hidden / 2))
    ]
    factors_short_name = [
        "factor_short_{0}".format(i) for i in range(0, int(hidden / 2))
    ]

    wfeatures = [
        f"{f}_{i}d" for f in window_data['features']
        for i in range(variant['window'])
    ]

    long_res = []
    short_res = []
    seq = 0
    for k, v in window_data['data'].groupby('time_stat'):
        seq += 1
        if seq < variant['seq_cycle']:
            continue
        print(k)
        array = window_data['data'].loc[k - variant['seq_cycle'] +
                                        1:k][wfeatures].values
        array = torch.from_numpy(array).reshape(
            -1, len(window_data['code'].unique()),
            len(window_data['features']) * variant['window'])
        array = array.transpose(1, 0)
        temporal_feature = array.unsqueeze(0).to(envoy.device)
        hidden_short, hidden_long = envoy.predict(start_time=0,
                                                  data=temporal_feature,
                                                  is_trans=False)

        hidden_short_features = pd.DataFrame(
            hidden_short,
            index=window_data['data'].loc[k][wfeatures].index,
            columns=factors_short_name)

        hidden_long_features = pd.DataFrame(
            hidden_long,
            index=window_data['data'].loc[k][wfeatures].index,
            columns=factors_long_name)

        short_res.append(hidden_short_features)
        long_res.append(hidden_long_features)

    long_features = pd.concat(long_res)
    short_features = pd.concat(short_res)

    long_filename = os.path.join(
        base_path, variant['universe'],
        "{0}_long_envoy0002_{1}c_{2}w.feather".format(variant['method'],
                                                      variant['seq_cycle'],
                                                      variant['window']))

    short_filename = os.path.join(
        base_path, variant['universe'],
        "{0}_short_envoy0002_{1}c_{2}w.feather".format(variant['method'],
                                                       variant['seq_cycle'],
                                                       variant['window']))
    long_features.reset_index().to_feather(long_filename)
    short_features.reset_index().to_feather(short_filename)


def create_mae_data(variant):
    window = 2
    pdb.set_trace()
    window_data = load_mae_misro(method=variant['method'],
                                 window=window,
                                 universe=variant['universe'])

    envoy = Liaison0004(id="dubhe_{0}_{1}".format(variant['method'],
                                                  variant['universe']),
                        features=window_data['features'],
                        targets=window_data['targets'],
                        ticker_count=len(window_data['code'].unique()),
                        window=window,
                        is_load=True)
    hidden_name = [
        "hidden_{0}".format(i) for i in range(0, envoy.dimension * 2)
    ]
    hidden_out = []
    for k, v in window_data['data'].groupby('trade_date'):
        print(k)
        pdb.set_trace()
        output = envoy.predict(v.values[np.newaxis, :, :])
        ar = np.squeeze(output[0].detach().cpu(), axis=0)
        ar = np.concatenate([ar, ar], axis=1)
        hidden = pd.DataFrame(ar, index=v.index, columns=hidden_name)
        hidden_out.append(hidden)

    hidden_data = pd.concat(hidden_out)
    filename = os.path.join(
        base_path, variant['universe'],
        "{0}_liaison0004_hidden.feather".format(variant['method']))

    hidden_data.reset_index().to_feather(filename)


def standard_data(variant):
    ## 加载 时序数据
    long_filename = os.path.join(
        base_path, variant['universe'],
        "{0}_long_envoy0002_{1}c_{2}w.feather".format(variant['method'],
                                                      variant['seq_cycle'], 3))

    short_filename = os.path.join(
        base_path, variant['universe'],
        "{0}_short_envoy0002_{1}c_{2}w.feather".format(variant['method'],
                                                       variant['seq_cycle'],
                                                       3))

    long_temporal_data = pd.read_feather(long_filename)
    short_temporal_data = pd.read_feather(short_filename)

    ## 加载 MAE数据
    mae_filename = os.path.join(
        base_path, variant['universe'],
        "{0}_liaison0004_hidden.feather".format(variant['method']))
    mae_temporal_data = pd.read_feather(mae_filename)

    ### MAE 数据  long 数据， short 数据
    total_data = mae_temporal_data.merge(long_temporal_data,
                                         on=['trade_date', 'code']).merge(
                                             short_temporal_data,
                                             on=['trade_date', 'code'])

    ### 标准化
    res = []
    total_data = total_data.sort_values(['trade_date', 'code'])
    features = [
        i for i in total_data.columns if i not in ['trade_date', 'code']
    ]
    total_data = total_data.set_index(['trade_date', 'code']).unstack()

    for ff in features:
        print(ff)
        f = alk_standardize(alk_winsorize(total_data[ff])).unstack()
        f[np.isnan(f)] = 0
        f.name = ff
        res.append(f)
    dimension_data = pd.concat(res, axis=1)
    filename = os.path.join(
        base_path, variant['universe'],
        "{0}_model_factors_normal.feather".format(variant['method']))
    dimension_data.reset_index().to_feather(filename)


def create_data(variant):
    create_timing_data(variant)
    create_mae_data(variant)
    ## 标准化
    #standard_data(variant)


def train(variant):
    batch_size = 64
    train_dataset, val_dataset, mae_features, long_temporal_features, short_temporal_features = load_micro(
        method=variant['method'],
        window=variant['window'],
        seq_cycle=variant['seq_cycle'],
        universe=variant['universe'],
        horizon=variant['horizon'])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    pawn = Pawn0003(id="dubhe_{0}_{1}_{2}h".format(variant['method'],
                                                   variant['universe'],
                                                   variant['horizon']),
                    features=train_dataset.features,
                    ticker_count=len(train_dataset.code.unique()),
                    window=variant['window'],
                    relational_dim=len(mae_features),
                    long_dim=len(long_temporal_features),
                    short_dim=len(short_temporal_features))

    max_episode = 110
    pawn._veatures_atten2.train_model(train_loader=train_loader,
                                      val_loader=val_loader,
                                      is_state_dict=True,
                                      model_dir=g_train_path,
                                      tb_dir=g_tensorboard_path,
                                      push_dir=g_push_path,
                                      epochs=max_episode)


def predict(variant):
    window_val_data, mae_features, long_temporal_features, short_temporal_features = load_misro(
        method=variant['method'],
        window=variant['window'],
        universe=variant['universe'],
        horizon=variant['horizon'])

    pawn = Pawn0003(id="dubhe_{0}_{1}_{2}h".format(variant['method'],
                                                   variant['universe'],
                                                   variant['horizon']),
                    features=window_val_data['features'],
                    ticker_count=len(window_val_data['code'].unique()),
                    window=variant['window'],
                    relational_dim=len(mae_features),
                    long_dim=len(long_temporal_features),
                    short_dim=len(short_temporal_features),
                    is_load=True)
    factors_name = ["factor_{0}".format(i) for i in range(0, 1)]

    hidden_name = ["hidden_{0}".format(i) for i in range(0, 256)]

    res = []

    # torch.Size([64, 532, 256])
    for k, v in window_val_data['data'].groupby('trade_date'):
        print(k)
        pdb.set_trace()
        output = pawn.predict(
            to_device(torch.from_numpy(v.values[np.newaxis, :, :])), False)

        factros = pd.DataFrame(np.squeeze(output[-1].detach().cpu(), axis=0),
                               index=v.index,
                               columns=factors_name)

        hidden = pd.DataFrame(np.squeeze(output[0].detach().cpu(), axis=0),
                              index=v.index,
                              columns=hidden_name)

        res.append(pd.concat([factros, hidden], axis=1))

    factors_data = pd.concat(res)

    filename = os.path.join(
        base_path, variant['universe'],
        "pawn0003_factors_{0}.feather".format(variant['horizon']))
    factors_data.reset_index().to_feather(filename)


def metrics(variant):
    filename = os.path.join(
        base_path, variant['universe'],
        "pawn0003_factors_{0}.feather".format(variant['horizon']))

    print(filename)
    factors_data = pd.read_feather(filename)
    pdb.set_trace()
    begin_date = factors_data['trade_date'].min().strftime('%Y-%m-%d')
    end_date = factors_data['trade_date'].max().strftime('%Y-%m-%d')
    remain_data = fetch_f1r_oo(begin_date, end_date, variant['universe'])
    dummy_fst = remain_data['dummy_test_f1r_open'] * remain_data[
        variant['universe']]
    yields_data = remain_data['ret_f1r_oo'] * remain_data[variant['universe']]
    factors_data = factors_data.set_index(['trade_date', 'code'])
    factor_columns = factors_data.columns
    res = []

    dirs = os.path.join(
        base_path, variant['universe'], 'metrics',
        'pawn0003_{0}_{1}'.format('factors', variant['horizon']))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    print(dirs)
    for i, col in enumerate(factor_columns):
        print("factor name:{0}".format(col))
        factors_data0 = factors_data[col].unstack() * dummy_fst
        yields_data0 = yields_data.reindex(factors_data0.index,
                                           columns=factors_data0.columns)
        dummy_fst0 = dummy_fst.reindex(factors_data0.index,
                                       columns=factors_data0.columns)
        st0 = kdmetrics(dummy_fst=dummy_fst0,
                        yields_data=yields_data0,
                        factor_data=factors_data0,
                        name=col)
        res.append(st0)
        if i % 50 == 0 and i != 0:
            results = pd.DataFrame(res)
            results.to_csv(os.path.join(dirs, 'metrics_{0}.csv'.format(i)),
                           encoding='utf-8')
            res = []
    if len(res) > 0:
        results = pd.DataFrame(res)
        results.to_csv(os.path.join(dirs, 'metrics_{0}.csv'.format(i)),
                       encoding='utf-8')
    print('done')


def polymer(variant):
    file_path = os.path.join(
        base_path, variant['universe'], 'metrics',
        'pawn0003_{0}_{1}'.format('factors', variant['horizon']))
    print(file_path)

    files = os.listdir(file_path)
    res = []
    for file in files:
        print(file)
        if file.endswith('.csv'):
            ms1 = pd.read_csv(os.path.join(file_path, file), index_col=0)
            res.append(ms1)
    metrics1 = metrics1 = pd.concat(res).sort_values(by=['sharp'],
                                                     ascending=False)
    metrics1 = metrics1.drop([
        'count_series',
        'returns_series',
        'ic_series',
        'turnover_series',
    ],
                             axis=1)[[
                                 'name', 'sharp', 'returns_mean',
                                 'returns_std', 'returns_mdd', 'fitness', 'ir',
                                 'ret2tv', 'turnover', 'maxdd'
                             ]]
    dirs = os.path.join(base_path, variant['universe'], 'metrics', 'perf')
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    metrics1.to_csv(os.path.join(dirs,
                                 'metrics_{0}.csv'.format(variant['horizon'])),
                    encoding='utf-8')


def show(variant):
    dirs = os.path.join(base_path, variant['universe'], 'metrics', 'perf')
    res = []
    for file in os.listdir(dirs):
        print(file)
        if file.endswith('.csv'):
            ms1 = pd.read_csv(os.path.join(dirs, file), index_col=0)
            name = file.split('_')[1].split('.')[0]
            ms1['name'] = ms1['name'].str.replace('hidden',
                                                  '{0}Factor'.format(name))
            res.append(ms1)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='sicro')
    parser.add_argument('--window', type=int, default=1)

    parser.add_argument('--horizon', type=int, default=5)

    parser.add_argument('--seq_cycle', type=int, default=15)
    parser.add_argument('--universe', type=str, default='hs300')

    args = parser.parse_args()

    #create_data(vars(args))
    train(vars(args))
    predict(vars(args))
    metrics(vars(args))
    polymer(vars(args))
    #show(vars(args))
