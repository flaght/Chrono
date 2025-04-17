import sys, os, torch, pdb, argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv


load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from ultron.optimize.wisem import *
from kdutils.data import fetch_base
from kdutils.kdmetrics import long_metrics

from kichaos.utils.env import *
from kichaos.datasets import CogniDataSet9
from kichaos.agent import Envoy0003

def load_misro(method, window, seq_cycle, horizon, time_format='%Y-%m-%d'):
    val_filename = os.path.join(os.environ['BASE_PATH'], method, 'evolution', str(1),
                                "val_model_normal.feather")
    val_data = pd.read_feather(val_filename).rename(columns={'trade_date':'trade_time'})

    nxt1_columns = val_data.filter(regex="^nxt1_").columns.to_list()

    columns = [
        col for col in val_data.columns
        if col not in ['trade_time', 'code'] + nxt1_columns
    ]
    val_data = val_data[['trade_time', 'code'] + columns +
                            ['nxt1_ret_{0}h'.format(horizon)]].sort_values(
                                ['trade_time', 'code'])
    val_data.rename(columns={'nxt1_ret_{0}h'.format(horizon): 'nxt1_ret'},
                      inplace=True)

    features = [
        col for col in val_data.columns
        if col not in ['trade_time', 'code', 'dummy', 'nxt1_ret']
    ]

    window_val_data = CogniDataSet3.create_windows(data=val_data.set_index(
        ['trade_time', 'code']),
                                                   features=features,
                                                   window=window,
                                                   time_name='trade_time')
    return window_val_data

def load_micro(method, window, seq_cycle, horizon, time_format='%Y-%m-%d'):
    pdb.set_trace()
    train_filename = os.path.join(
        os.environ['BASE_PATH'], method, 'evolution', str(1), #str(horizon),
        "train_model_normal.feather")
    train_data = pd.read_feather(train_filename).rename(columns={'trade_date':'trade_time'})
    pdb.set_trace()
    val_filename = os.path.join(os.environ['BASE_PATH'], method,'evolution', str(1),#str(horizon),
                                "val_model_normal.feather")
    val_data = pd.read_feather(val_filename).rename(columns={'trade_date':'trade_time'})

    nxt1_columns = train_data.filter(regex="^nxt1_").columns.to_list()

    columns = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code'] + nxt1_columns
    ]
    pdb.set_trace()
    train_data = train_data[['trade_time', 'code'] + columns +
                            ['nxt1_ret_{0}h'.format(horizon)]].sort_values(
                                ['trade_time', 'code'])

    val_data = val_data[['trade_time', 'code'] + columns +
                            ['nxt1_ret_{0}h'.format(horizon)]].sort_values(
                                ['trade_time', 'code'])

    train_data.rename(columns={'nxt1_ret_{0}h'.format(horizon): 'nxt1_ret'},
                      inplace=True)
    val_data.rename(columns={'nxt1_ret_{0}h'.format(horizon): 'nxt1_ret'},
                      inplace=True)

    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code', 'dummy', 'nxt1_ret']
    ]
    pdb.set_trace()
    train_dataset = CogniDataSet9.generate(train_data,
                                           seq_cycle=seq_cycle,
                                           features=features,
                                           window=window,
                                           target=['nxt1_ret'],
                                           time_name='trade_time',
                                           time_format=time_format)
    val_dataset = CogniDataSet9.generate(val_data,
                                         seq_cycle=seq_cycle,
                                         features=features,
                                         window=window,
                                         target=['nxt1_ret'],
                                         time_name='trade_time',
                                         time_format=time_format)
    return train_dataset, val_dataset, train_data, val_data



def train(variant):
    batch_size = 256
    train_dataset, val_dataset, _, _ = load_micro(
        method=variant['method'],
        window=variant['window'],
        seq_cycle=variant['seq_cycle'],
        horizon=variant['horizon'])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    pdb.set_trace()
    envoy = Envoy0003(id="dubhe_{0}_{1}c".format(variant['method'],
                                                     variant['seq_cycle']),
                      features=train_dataset.features,
                      ticker_count=len(train_dataset.code.unique()),
                      window=variant['window'],
                      seq_cycle=variant['seq_cycle'],
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
    window_data = load_misro(method=variant['method'],
                             window=variant['window'],
                             seq_cycle=variant['seq_cycle'],
                             horizon=variant['horizon'])
    envoy = Envoy0003(id="dubhe_{0}_{1}c".format(variant['method'],
                                                     variant['seq_cycle']),
                      features=window_data['features'],
                      ticker_count=len(window_data['code'].unique()),
                      window=variant['window'],
                      is_debug=True)

    envoy._create_custom_transient_hybrid_transformer(model_path=g_push_path,
                                                      id="{0}".format(
                                                          variant['horizon']))

    hidden = envoy._custom_transient_hybrid_transformer.hidden_size()
    factors_name = ["hidden_{0}".format(i) for i in range(0, hidden)]

    wfeatures = [
        f"{f}_{i}d" for f in window_data['features']
        for i in range(variant['window'])
    ]
    ticker_count = len(window_data['code'].unique())
    envoy._custom_transient_hybrid_transformer.eval()

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
        with torch.no_grad():
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
            columns=['factor'])
        data = pd.concat([hidden_features, outputs], axis=1)
        res.append(data)
    pdb.set_trace()
    factors_data = pd.concat(res)

    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'Timing', envoy.kid)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    pdb.set_trace()
    filename = os.path.join(dirs, "envoy0003_factors.feather")
    factors_data.reset_index().to_feather(filename)

def metrics(variant):
    pdb.set_trace()
    path1 = 'envoy0003_transient_hybrid_transformer_3p_2s_ranking_5h_dubhe_hs300_10c'
    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'Timing', 
            path1)
    
    filename = os.path.join(dirs, "envoy0003_factors.feather")
    factors_data = pd.read_feather(filename)
    
    begin_date = factors_data['trade_time'].min().strftime('%Y-%m-%d')
    end_date = factors_data['trade_time'].max().strftime('%Y-%m-%d')
    remain_data = fetch_base(begin_date, end_date)
    ret_data = remain_data['ret_f1r_cc']
    dummy120_fst = remain_data['dummy120_fst']
    dummy120_fst_close = remain_data['dummy120_fst_close']
    hs300 = remain_data['hs300']
    zz1000 = remain_data['zz1000']
    zz500 = remain_data['zz500']

    yields_data = ret_data.reindex(dummy120_fst.index, columns=dummy120_fst.columns)
    yields_data = yields_data[(hs300 == 1) | (zz500 == 1) | (zz1000 == 1)] * dummy120_fst_close * dummy120_fst
    dummy_fst = dummy120_fst_close * dummy120_fst
    factors_data = factors_data.set_index(['trade_time', 'code'])
    factor_columns = factors_data.columns
    res = []
    pdb.set_trace()
    metric_dir = os.path.join(dirs, 'metrics')
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    pdb.set_trace()
    for i, col in enumerate(factor_columns):
        print("factor name:{0}".format(col))
        factors_data0 = factors_data[col].copy()
        factors_data0 = factors_data0.unstack()
        factors_data0 = factors_data0.reindex(dummy120_fst.index, columns=dummy120_fst.columns)
        factors_data0 = factors_data0[(hs300 == 1) | (zz500 == 1) | (zz1000 == 1)] * dummy_fst

        yields_data0 = yields_data.reindex(factors_data0.index,
                                           columns=factors_data0.columns)
        dummy_fst0 = dummy_fst.reindex(factors_data0.index,
                                       columns=factors_data0.columns)
        
        st0 = long_metrics(dummy_fst=dummy_fst0, yields_data=yields_data0,
                    factor_data=factors_data0, name=col)
        res.append(st0)

        if i % 50 == 0 and i != 0:
            results = pd.DataFrame(res)
            results.to_csv(os.path.join(metric_dir, 'metrics_{0}.csv'.format(i)),
                           encoding='utf-8')
            res = []
    if len(res) > 0:
        results = pd.DataFrame(res)
        results.to_csv(os.path.join(metric_dir, 'metrics_{0}.csv'.format(i)),
                       encoding='utf-8')


def polymer(variant):
    path1 = 'envoy0003_transient_hybrid_transformer_3p_2s_ranking_5h_dubhe_hs300_10c'
    file_path = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'Timing', path1, 'metrics')
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
                             axis=1)
    pdb.set_trace()
    metrics1 = metrics1[['name', 'sharp', 'returns_mean',
                                 'returns_std', 'returns_mdd', 'fitness', 'ir',
                                 'ret2tv', 'turnover', 'maxdd'
                             ]]
    pdb.set_trace()
    print(metrics1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default=os.environ['DUMMY_NAME'])
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--seq_cycle', type=int, default=5)

    args = parser.parse_args()

    train(vars(args))
    #predict(vars(args))
    #metrics(vars(args))
    #polymer(vars(args))