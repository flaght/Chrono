import sys, os, torch, pdb, argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from ultron.optimize.wisem import *

load_dotenv()
sys.path.append('../../kichaos')

from kichaos.datasets import CogniDataSet3
from kichaos.agent.envoy import Envoy0002
from kichaos.utils.env import *

from kdutils.macro import base_path
from kdutils.data import fetch_f1r_oo
from kdutils.metrics import metrics as kdmetrics

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


def load_misro(method,
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
    pdb.set_trace()
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
    pos = int(len(dates) * 0.7)
    train_data = total_data[total_data['trade_date'].isin(dates[:pos])]
    val_data = total_data[total_data['trade_date'].isin(dates[pos - window -
                                                              seq_cycle + 1:])]

    features = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code', 'dummy', 'nxt1_ret']
    ]
    window_val_data = CogniDataSet3.create_windows(data=val_data.set_index(
        ['trade_date', 'code']),
                                                   features=features,
                                                   window=window,
                                                   time_name='trade_date')
    return window_val_data


def load_micro(method,
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
    pos = int(len(dates) * 0.7)
    train_data = total_data[total_data['trade_date'].isin(dates[:pos])]
    val_data = total_data[total_data['trade_date'].isin(dates[pos:])]

    features = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code', 'dummy', 'nxt1_ret']
    ]

    train_dataset = CogniDataSet3.generate(train_data,
                                           seq_cycle=seq_cycle,
                                           features=features,
                                           window=window,
                                           target=['nxt1_ret'],
                                           time_name='trade_date',
                                           time_format=time_format)
    val_dataset = CogniDataSet3.generate(val_data,
                                         seq_cycle=seq_cycle,
                                         features=features,
                                         window=window,
                                         target=['nxt1_ret'],
                                         time_name='trade_date',
                                         time_format=time_format)
    return train_dataset, val_dataset, train_data, val_data


def train(variant):
    batch_size = 16
    train_dataset, val_dataset, _, _ = load_micro(
        method=variant['method'],
        window=variant['window'],
        seq_cycle=variant['seq_cycle'],
        horizon=variant['horizon'],
        universe=variant['universe'])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    envoy = Envoy0002(id="dubhe_{0}_{1}c_{2}".format(variant['method'],
                                                     variant['seq_cycle'],
                                                     variant['universe']),
                      features=train_dataset.features,
                      ticker_count=len(train_dataset.code.unique()),
                      window=variant['window'],
                      is_debug=True)

    envoy._create_custom_transient_hybrid_transformer(model_path=None,
                                                      id="{0}".format(
                                                          variant['horizon']))

    max_episode = 100
    envoy._custom_transient_hybrid_transformer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        is_state_dict=True,
        model_dir=envoy.train_path,
        tb_dir=envoy.tensorboard_path,
        push_dir=envoy.push_path,
        epochs=max_episode)


def predict(variant):
    pdb.set_trace()
    window_data = load_misro(method=variant['method'],
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
                      is_debug=True)

    envoy._create_custom_transient_hybrid_transformer(model_path=g_push_path,
                                                      id="{0}".format(
                                                          variant['horizon']))
    pdb.set_trace()
    hidden = envoy._custom_transient_hybrid_transformer.hidden_size()
    factors_name = ["factor_{0}".format(i) for i in range(0, hidden)]

    wfeatures = [
        f"{f}_{i}d" for f in window_data['features']
        for i in range(variant['window'])
    ]

    res = []
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
        _, hidden_features, outputs = envoy._custom_transient_hybrid_transformer.predict(
            temporal_feature, False)

        hidden_features = hidden_features.squeeze(
            0).detach().cpu().numpy().reshape(
                len(window_data['code'].unique()), -1)

        hidden_features = pd.DataFrame(
            hidden_features,
            index=window_data['data'].loc[k][wfeatures].index,
            columns=factors_name)
        outputs = pd.DataFrame(
            outputs.squeeze().detach().cpu(),
            index=window_data['data'].loc[k][wfeatures].index,
            columns=['output'])
        data = pd.concat([hidden_features, outputs], axis=1)
        res.append(data)

    factors_data = pd.concat(res)

    dirs = os.path.join(base_path, variant['universe'], envoy.kid)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(
        dirs,
        "envoy0002_{0}h_{1}c_{2}w.feather".format(variant['horizon'],
                                                  variant['seq_cycle'],
                                                  variant['window']))
    factors_data.reset_index().to_feather(filename)


def metrics(variant):
    basic_path = 'envoy0002_transient_hybrid_transformer_3p_2s_ranking_5h_dubhe_sicro_10c_hs300'
    dirs = os.path.join(base_path, variant['universe'], basic_path)
    filename = os.path.join(
        dirs,
        "envoy0002_{0}h_{1}c_{2}w.feather".format(variant['horizon'],
                                                  variant['seq_cycle'],
                                                  variant['window']))

    factors_data = pd.read_feather(filename)
    begin_date = factors_data['trade_date'].min().strftime('%Y-%m-%d')
    end_date = factors_data['trade_date'].max().strftime('%Y-%m-%d')
    remain_data = fetch_f1r_oo(begin_date, end_date, variant['universe'])
    dummy_fst = remain_data['dummy_test_f1r_open'] * remain_data[
        variant['universe']]
    yields_data = remain_data['ret_f1r_oo'] * remain_data[variant['universe']]
    factors_data = factors_data.set_index(['trade_date', 'code'])
    factor_columns = factors_data.columns
    res = []
    metrics_dirs = os.path.join(dirs, 'metrics')
    if not os.path.exists(metrics_dirs):
        os.makedirs(metrics_dirs)
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
            results.to_csv(os.path.join(metrics_dirs,
                                        'metrics_{0}.csv'.format(i)),
                           encoding='utf-8')
            res = []
    if len(res) > 0:
        results = pd.DataFrame(res)
        results.to_csv(os.path.join(metrics_dirs, 'metrics_{0}.csv'.format(i)),
                       encoding='utf-8')


def polymer(variant):
    basic_path = 'envoy0002_transient_hybrid_transformer_3p_2s_ranking_5h_dubhe_sicro_10c_hs300'
    dirs = os.path.join(base_path, variant['universe'], basic_path)
    file_path = os.path.join(
        dirs, 'metrics')

    files = os.listdir(file_path)
    res = []
    for file in files:
        print(file)
        if file.endswith('.csv'):
            ms1 = pd.read_csv(os.path.join(file_path, file), index_col=0)
            res.append(ms1)
    metrics1 = metrics1 = pd.concat(res).sort_values(by=['sharp'],
                                                     ascending=False)
    metrics1 = metrics1.drop(
        ['count_series', 'returns_series', 'ic_series', 'turnover_series'],
        axis=1)[[
            'name', 'sharp', 'returns_mean', 'returns_std', 'returns_mdd',
            'fitness', 'ir', 'ret2tv', 'turnover', 'maxdd'
        ]]
    metrics1
    pdb.set_trace()
    print()


## 加载配对模型
def load_pairs(variant):
    window_data = load_misro(method=variant['method'],
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
    pdb.set_trace()
    print('-->')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='sicro')
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--seq_cycle', type=int, default=10)
    parser.add_argument('--universe', type=str, default='hs300')

    args = parser.parse_args()

    train(vars(args))
    #predict(vars(args))
    #metrics(vars(args))
    #polymer(vars(args))
    #load_pairs(vars(args))
