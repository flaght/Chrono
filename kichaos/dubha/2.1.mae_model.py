import sys, os, torch, pdb, argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


from ultron.optimize.wisem import *
from kdutils.data import fetch_base
from kdutils.kdmetrics import long_metrics

from kichaos.datasets import CogniDataSet6
from kichaos.agent import Liaison0004


def load_misro(method, horizon, window,time_format='%Y-%m-%d'):
    val_filename = os.path.join(os.environ['BASE_PATH'], method,'evolution', str(horizon),
                                "val_model_normal.feather")
    val_data = pd.read_feather(val_filename).rename(columns={'trade_date':'trade_time'})
    nxt1_columns = val_data.filter(regex="^nxt1_").columns.to_list()
    features = [
        col for col in val_data.columns
        if col not in ['trade_time', 'code', 'chg_pct'] + nxt1_columns
    ]

    window_val_data = CogniDataSet6.create_windows(data=val_data.set_index(
        ['trade_time', 'code']),
                                                   features=features,
                                                   target=nxt1_columns,
                                                   window=window,
                                                   time_name='trade_time')
    return window_val_data


def load_micro(method,horizon,
               window=3,
               time_format='%Y-%m-%d'):
    
    train_filename = os.path.join(
        os.environ['BASE_PATH'], method,'evolution', str(horizon),
        "train_model_normal.feather")
    train_data = pd.read_feather(train_filename).rename(columns={'trade_date':'trade_time'})

    val_filename = os.path.join(os.environ['BASE_PATH'], method,'evolution', str(horizon),
                                "val_model_normal.feather")
    val_data = pd.read_feather(val_filename).rename(columns={'trade_date':'trade_time'})

    nxt1_columns = train_data.filter(regex="^nxt1_").columns.to_list()
    columns = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code'] + nxt1_columns
    ]
    train_data = train_data[['trade_time', 'code'] + columns +
                            nxt1_columns].sort_values(['trade_time', 'code'])
    val_data = val_data[['trade_time', 'code'] + columns +
                        nxt1_columns].sort_values(['trade_time', 'code'])

    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code', 'chg_pct'] + nxt1_columns
    ]
    pdb.set_trace()
    train_dataset = CogniDataSet6.generate(
        train_data,
        codes=train_data.code.unique().tolist(),
        features=features,
        window=window,
        target=nxt1_columns,
        time_name='trade_time',
        time_format=time_format)
    
    val_dataset = CogniDataSet6.generate(val_data,
                                         codes=val_data.code.unique().tolist(),
                                         features=features,
                                         window=window,
                                         target=nxt1_columns,
                                         time_name='trade_time',
                                         time_format=time_format)
    return train_dataset, val_dataset

def train(variant):
    batch_size = 16
    train_dataset, val_dataset = load_micro(
        method=variant['method'],
        horizon=variant['horizon'],
        window=variant['window'])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    pdb.set_trace()
    envoy = Liaison0004(id="dubhe_{0}_{1}".format(variant['method'],variant['horizon']),
                        features=train_dataset.features,
                        targets=train_dataset.target,
                        ticker_count=len(train_dataset.code.unique()),
                        window=variant['window'])
    max_episode = 100
    pdb.set_trace()
    envoy._planar_hybrid_transformer.train_model(train_loader=train_loader,
                                                 val_loader=val_loader,
                                                 is_state_dict=True,
                                                 model_dir=envoy.train_path,
                                                 tb_dir=envoy.tensorboard_path,
                                                 push_dir=envoy.push_path,
                                                 epochs=max_episode)

def predict(variant):
    pdb.set_trace()
    window_val_data = load_misro(method=variant['method'],window=variant['window'], horizon=variant['horizon'])
    pdb.set_trace()
    envoy = Liaison0004(id="dubhe_{0}_{1}".format(variant['method'],variant['horizon']),
                        features=window_val_data['features'],
                        targets=window_val_data['targets'],
                        ticker_count=len(window_val_data['code'].unique()),
                        window=variant['window'],is_load=True)
    
    factors_name = [
        "factor_{0}".format(i) for i in range(0, len(window_val_data['targets']))
    ]
    hidden_name = ["hidden_{0}".format(i) for i in range(0, envoy.dimension)]
    res1_out = []
    res2_out = []
    pdb.set_trace()
    for k, v in window_val_data['data'].groupby('trade_time'):
        print(k)
        output = envoy.predict(v.values[np.newaxis, :, :])
        factros = pd.DataFrame(np.squeeze(output[-1].detach().cpu(), axis=0),
                               index=v.index,
                               columns=factors_name)
        res1_out.append(factros)

        hidden = pd.DataFrame(np.squeeze(output[0].detach().cpu(), axis=0),
                              index=v.index,
                              columns=hidden_name)
        res2_out.append(hidden)

    factors_data = pd.concat(res1_out)
    hidden_data = pd.concat(res2_out)
    
    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'MAE1', envoy.kid)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    pdb.set_trace()
    filename = os.path.join(dirs, "liaison0004_factors_{0}.feather".format(variant['horizon']))
    factors_data.reset_index().drop(['time_stat'],axis=1).to_feather(filename)
    pdb.set_trace()
    filename = os.path.join(dirs, "liaison0004_hidden_{0}.feather".format(variant['horizon']))
    hidden_data.reset_index().drop(['time_stat'],axis=1).to_feather(filename)


def metrics(variant):
    pdb.set_trace()
    path1 = 'liaison0004_7p_3s_ranking_state_dubhe_hs300_1'
    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'MAE1', 
            path1)
    
    filename = os.path.join(dirs, "liaison0004_factors_{0}.feather".format(variant['horizon']))
    factors_data = pd.read_feather(filename)

    filename = os.path.join(dirs, "liaison0004_hidden_{0}.feather".format(variant['horizon']))
    hidden_data = pd.read_feather(filename)
    factors_data = factors_data.merge(hidden_data,
                                      on=['trade_time', 'code'],
                                      how='left')
    
    begin_date = factors_data['trade_time'].min().strftime('%Y-%m-%d')
    end_date = factors_data['trade_time'].max().strftime('%Y-%m-%d')
    remain_data = fetch_base(begin_date, end_date)
    ret_data = remain_data['ret_f1r_cc']
    dummy120_fst = remain_data['dummy120_fst']
    dummy120_fst_close = remain_data['dummy120_fst_close']
    hs300 = remain_data['hs300']
    #zz1000 = remain_data['zz1000']
    #zz500 = remain_data['zz500']

    yields_data = ret_data.reindex(dummy120_fst.index, columns=dummy120_fst.columns)
    #yields_data = yields_data[(hs300 == 1) | (zz500 == 1) | (zz1000 == 1)] * dummy120_fst_close * dummy120_fst
    yields_data = yields_data[(hs300 == 1)] * dummy120_fst_close * dummy120_fst
    dummy_fst = dummy120_fst_close * dummy120_fst
    factors_data = factors_data.set_index(['trade_time', 'code'])
    factor_columns = factors_data.columns
    res = []
    pdb.set_trace()
    metric_dir = os.path.join(dirs, 'metrics')
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    for i, col in enumerate(factor_columns):
        print("factor name:{0}".format(col))
        factors_data0 = factors_data[col].copy()
        factors_data0 = factors_data0.unstack()
        factors_data0 = factors_data0.reindex(dummy120_fst.index, columns=dummy120_fst.columns)
        #factors_data0 = factors_data0[(hs300 == 1) | (zz500 == 1) | (zz1000 == 1)] * dummy_fst
        factors_data0 = factors_data0[(hs300 == 1)] * dummy_fst
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
    path1 = 'liaison0004_6p_3s_ranking_state_dubhe_hs300_1'
    file_path = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'MAE1', path1, 'metrics')
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
    parser.add_argument('--horizon', type=int, default=1)


    args = parser.parse_args()

    #train(vars(args))
    #predict(vars(args))
    #metrics(vars(args))
    polymer(vars(args))