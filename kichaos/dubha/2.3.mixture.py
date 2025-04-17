import sys, os, torch, pdb, argparse, re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from kdutils.data import fetch_base
from kdutils.kdmetrics import long_metrics
from kichaos.utils.env import *
from kichaos.datasets import CogniDataSet6
from kichaos.agent import Liaison0004
from kichaos.datasets import CogniDataSet3
from kichaos.agent import Envoy0003
from kichaos.datasets import CogniDataSet8
from kichaos.agent.pawn import Pawn0004
from ultron.optimize.wisem import *
from ultron.strategy.models.processing import winsorize as alk_winsorize
from ultron.strategy.models.processing import standardize as alk_standardize

def mae_load_misro(method, window,time_format='%Y-%m-%d'):
    train_filename = os.path.join(os.environ['BASE_PATH'], method, 'evolution',
                str(1), "train_model_normal.feather")
    val_filename = os.path.join(os.environ['BASE_PATH'], method,'evolution', str(1), 
                                "val_model_normal.feather")
    
    train_data = pd.read_feather(train_filename).rename(columns={'trade_date':'trade_time'})
    val_data = pd.read_feather(val_filename).rename(columns={'trade_date':'trade_time'})
    nxt1_columns = val_data.filter(regex="^nxt1_").columns.to_list()
    features = [
        col for col in val_data.columns
        if col not in ['trade_time', 'code', 'chg_pct'] + nxt1_columns
    ]

    window_train_data = CogniDataSet6.create_windows(data=train_data.set_index(
        ['trade_time', 'code']),
                                                   features=features,
                                                   target=nxt1_columns,
                                                   window=window,
                                                   time_name='trade_time')


    window_val_data = CogniDataSet6.create_windows(data=val_data.set_index(
        ['trade_time', 'code']),
                                                   features=features,
                                                   target=nxt1_columns,
                                                   window=window,
                                                   time_name='trade_time')
    return window_train_data,window_val_data

def create_mae_data(variant):
    window = 1
    window_train_data,window_val_data = mae_load_misro(method=variant['method'],window=window)
    envoy = Liaison0004(id="dubhe_{0}_{1}".format(variant['method'], 1),
                        features=window_val_data['features'],
                        targets=window_val_data['targets'],
                        ticker_count=len(window_val_data['code'].unique()),
                        window=variant['window'],is_load=True)
    hidden_name = ["mae_hidden_{0}".format(i) for i in range(0, envoy.dimension)]
    envoy._planar_hybrid_transformer.eval()
    res_train = []
    for k, v in window_train_data['data'].groupby('trade_time'):
        print(k)
        output = envoy.predict(v.values[np.newaxis, :, :])
        hidden = pd.DataFrame(np.squeeze(output[0].detach().cpu(), axis=0),
                              index=v.index,
                              columns=hidden_name)
        res_train.append(hidden)

    res_val = []
    for k, v in window_val_data['data'].groupby('trade_time'):
        print(k)
        output = envoy.predict(v.values[np.newaxis, :, :])
        hidden = pd.DataFrame(np.squeeze(output[0].detach().cpu(), axis=0),
                              index=v.index,
                              columns=hidden_name)
        res_val.append(hidden)

    mae_train_data = pd.concat(res_train)
    mae_val_data = pd.concat(res_val)

    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'mixture', envoy.kid)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    pdb.set_trace()
    filename = os.path.join(dirs, "train_data.feather")
    mae_train_data.reset_index().drop(['time_stat'],axis=1).to_feather(filename)

    filename = os.path.join(dirs, "val_data.feather")
    mae_val_data.reset_index().drop(['time_stat'],axis=1).to_feather(filename)

def timing_load_misro(method, window, seq_cycle, horizon, time_format='%Y-%m-%d'):
    train_filename = os.path.join(os.environ['BASE_PATH'], method,'evolution', str(1),
                                "train_model_normal.feather")
    val_filename = os.path.join(os.environ['BASE_PATH'], method,'evolution', str(1),
                                "val_model_normal.feather")
    
    train_data = pd.read_feather(train_filename).rename(columns={'trade_date':'trade_time'})
    val_data = pd.read_feather(val_filename).rename(columns={'trade_date':'trade_time'})
    train_data = train_data.sort_values(by=['trade_time','code'])
    val_data = val_data.sort_values(by=['trade_time','code'])
    nxt1_columns = val_data.filter(regex="^nxt1_").columns.to_list()
    features = [
        col for col in val_data.columns
        if col not in ['trade_time', 'code', 'chg_pct'] + nxt1_columns
    ]

    window_train_data = CogniDataSet3.create_windows(data=train_data.set_index(
        ['trade_time', 'code']),
                                                   features=features,
                                                   window=window,
                                                   time_name='trade_time')

    window_val_data = CogniDataSet3.create_windows(data=val_data.set_index(
        ['trade_time', 'code']),
                                                   features=features,
                                                   window=window,
                                                   time_name='trade_time')
    return window_train_data, window_val_data


def create_timings_data(method, seq_cycle, window, horizon, name, 
                    window_train_data, window_val_data):
    pdb.set_trace()
    envoy = Envoy0003(id="dubhe_{0}_{1}c".format(method,seq_cycle),
                      features=window_train_data['features'],
                      ticker_count=len(window_train_data['code'].unique()),
                      window=window,
                      is_debug=True)

    envoy._create_custom_transient_hybrid_transformer(model_path=g_push_path,
                                                      id="{0}".format(horizon))
    
    hidden = envoy._custom_transient_hybrid_transformer.hidden_size()
    factors_name = ["{0}_hidden_{1}".format(name, i) for i in range(0, hidden)]
    wfeatures = [
        f"{f}_{i}d" for f in window_train_data['features']
        for i in range(window)
    ]
    ticker_count = len(window_train_data['code'].unique())
    envoy._custom_transient_hybrid_transformer.eval()

    res_train = []
    seq = 0
    for k, v in window_train_data['data'].groupby('time_stat'):
        seq += 1
        if seq < seq_cycle:
            continue
        print(k)
        array = window_train_data['data'].loc[k - seq_cycle +
                                        1:k][wfeatures].values
        array = torch.from_numpy(array).reshape(
            -1, len(window_train_data['code'].unique()),
            len(window_train_data['features']) * window)
        array = array.transpose(1, 0)
        temporal_feature = array.unsqueeze(0).to(envoy.device)
        with torch.no_grad():
            _, hidden_features, outputs = envoy._custom_transient_hybrid_transformer.predict(
                temporal_feature, False)
        
        hidden_features = hidden_features.squeeze(
            0).detach().cpu().numpy().reshape(
                len(window_train_data['code'].unique()), -1)

        hidden_features = pd.DataFrame(
            hidden_features,
            index=window_train_data['data'].loc[k][wfeatures].index,
            columns=factors_name)

        res_train.append(hidden_features)

    factors_train_data = pd.concat(res_train)



    res_val = []
    seq = 0
    for k, v in window_val_data['data'].groupby('time_stat'):
        seq += 1
        if seq < seq_cycle:
            continue
        print(k)
        array = window_val_data['data'].loc[k - seq_cycle +
                                        1:k][wfeatures].values
        array = torch.from_numpy(array).reshape(
            -1, len(window_val_data['code'].unique()),
            len(window_val_data['features']) * window)
        array = array.transpose(1, 0)
        temporal_feature = array.unsqueeze(0).to(envoy.device)
        with torch.no_grad():
            _, hidden_features, outputs = envoy._custom_transient_hybrid_transformer.predict(
                temporal_feature, False)
        
        hidden_features = hidden_features.squeeze(
            0).detach().cpu().numpy().reshape(
                len(window_val_data['code'].unique()), -1)

        hidden_features = pd.DataFrame(
            hidden_features,
            index=window_val_data['data'].loc[k][wfeatures].index,
            columns=factors_name)

        res_val.append(hidden_features)

    factors_val_data = pd.concat(res_val)
    
    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'mixture', envoy.kid)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    filename = os.path.join(dirs, "train_data.feather")
    factors_train_data.reset_index().to_feather(filename)

    filename = os.path.join(dirs, "val_data.feather")
    factors_val_data.reset_index().to_feather(filename)

        
def create_timing_data(variant):
    window = 1
    seq_cycle = 10
    window_train_data, window_val_data = timing_load_misro(
        method=variant['method'], window=window, 
        seq_cycle=seq_cycle, horizon=1)

    create_timings_data(method=variant['method'], seq_cycle=seq_cycle, 
                    window=window, horizon=1, name='short', 
                    window_train_data=window_train_data, 
                    window_val_data=window_val_data)
    
    create_timings_data(method=variant['method'], seq_cycle=seq_cycle, 
                    window=window, horizon=3, name='medium', 
                    window_train_data=window_train_data, 
                    window_val_data=window_val_data)

    create_timings_data(method=variant['method'], seq_cycle=seq_cycle, 
                    window=window, horizon=5, name='long', 
                    window_train_data=window_train_data, 
                    window_val_data=window_val_data)

def standard_data(variant):
    pdb.set_trace()
    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'mixture')
    ### 加载MAE数据
    mae_name = "liaison0004_7p_3s_ranking_state_dubhe_{0}_1".format(os.environ['DUMMY_NAME'])
    mae_train_data = pd.read_feather(os.path.join(dirs, mae_name, 'train_data.feather'))
    mae_val_data = pd.read_feather(os.path.join(dirs, mae_name, 'val_data.feather'))

    ### 加载时序数据
    timing_name = "envoy0003_transient_hybrid_transformer_3p_2s_ranking_{}h_dubhe_{}_{}c"
    
    short_train_data = pd.read_feather(os.path.join(dirs, timing_name.format(1, os.environ['DUMMY_NAME'], 10), 'train_data.feather'))
    short_val_data = pd.read_feather(os.path.join(dirs, timing_name.format(1, os.environ['DUMMY_NAME'], 10), 'val_data.feather'))

    medium_train_data = pd.read_feather(os.path.join(dirs, timing_name.format(3, os.environ['DUMMY_NAME'], 10), 'train_data.feather'))
    medium_val_data = pd.read_feather(os.path.join(dirs, timing_name.format(3, os.environ['DUMMY_NAME'], 10), 'val_data.feather'))

    long_train_data = pd.read_feather(os.path.join(dirs, timing_name.format(5, os.environ['DUMMY_NAME'], 10), 'train_data.feather'))
    long_val_data = pd.read_feather(os.path.join(dirs, timing_name.format(5, os.environ['DUMMY_NAME'], 10), 'val_data.feather'))
    pdb.set_trace()
    train_data = mae_train_data.merge(short_train_data, on=['trade_time','code']).merge(
        medium_train_data, on=['trade_time','code']).merge(
            long_train_data, on=['trade_time','code'])

    val_data = mae_val_data.merge(short_val_data, on=['trade_time','code']).merge(
        medium_val_data, on=['trade_time','code']).merge(
            long_val_data, on=['trade_time','code'])
    
    ###dummy
    ### 标准化
    train_begin_date = train_data['trade_time'].min().strftime('%Y-%m-%d')
    train_end_date = train_data['trade_time'].max().strftime('%Y-%m-%d')

    val_begin_date = val_data['trade_time'].min().strftime('%Y-%m-%d')
    val_end_date = val_data['trade_time'].max().strftime('%Y-%m-%d')

    train_data_pd = fetch_base(train_begin_date, train_end_date)
    val_data_pd = fetch_base(val_begin_date, val_end_date)

    res_train = []
    res_val = []
    pdb.set_trace()
    features = [col for col in train_data.columns if col not in ['trade_time','code']]
    train_data = train_data.set_index(['trade_time','code']).unstack()
    val_data = val_data.set_index(['trade_time','code']).unstack()
    pdb.set_trace()
    for ff in features:
        print(ff)
        train_f = train_data[ff]
        train_f = train_f.reindex(index=train_data_pd['dummy120_fst'].index, columns=train_data_pd['dummy120_fst'].columns)
        #train_f = train_f[(train_data_pd['hs300']==1)|(
        #    train_data_pd['zz500']==1)|(
        #        train_data_pd['zz1000']==1)] * train_data_pd['dummy120_fst_close'] * train_data_pd['dummy120_fst']
        #train_f = train_f[(train_data_pd['hs300']==1)] * train_data_pd['dummy120_fst_close'] * train_data_pd['dummy120_fst']
        train_f = alk_standardize(alk_winsorize(train_f))
        train_f = train_f.stack()
        
        val_f = val_data[ff]
        val_f = val_f.reindex(index=val_data_pd['dummy120_fst'].index, columns=val_data_pd['dummy120_fst'].columns)
        #val_f = val_f[(val_data_pd['hs300']==1)|(
        #    val_data_pd['zz500']==1)|(
        #        val_data_pd['zz1000']==1)] * val_data_pd['dummy120_fst_close'] * val_data_pd['dummy120_fst']
        #val_f = val_f[(val_data_pd['hs300']==1)] * val_data_pd['dummy120_fst_close'] * val_data_pd['dummy120_fst']
        val_f = alk_standardize(alk_winsorize(val_f))
        val_f = val_f.stack()

        train_f.name = ff
        val_f.name = ff


        res_train.append(train_f)
        res_val.append(val_f)
    pdb.set_trace()
    train_dimension_data = pd.concat(res_train,axis=1)
    val_dimension_data = pd.concat(res_val,axis=1)

    train_dimension_data = train_dimension_data.unstack().fillna(method='ffill').fillna(0)
    train_dimension_data = train_dimension_data.stack()

    val_dimension_data = val_dimension_data.unstack().fillna(method='ffill').fillna(0)
    val_dimension_data = val_dimension_data.stack()

    filename = os.path.join(dirs, "normal_train_factors.feather")
    train_dimension_data.reset_index().to_feather(filename)

    filename = os.path.join(dirs, "normal_val_factors.feather")
    val_dimension_data.reset_index().to_feather(filename)
    print('-->')

def build_data(variant):
    pdb.set_trace()
    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'mixture')
    ### 读取特征
    train_factors_data = pd.read_feather(os.path.join(dirs, "normal_train_factors.feather"))
    val_factors_data = pd.read_feather(os.path.join(dirs, "normal_val_factors.feather"))

    ### 读取收益率
    filename = os.path.join(os.environ['BASE_PATH'], variant['method'],
                            "normal_yields.feather")
    normal_yields_data = pd.read_feather(filename)

    train_data = train_factors_data.merge(normal_yields_data, on=['trade_date','code'])

    val_data = val_factors_data.merge(normal_yields_data, on=['trade_date','code'])

    filename = os.path.join(dirs, "normal_train.feather")
    train_data.reset_index(drop=True).to_feather(filename)

    filename = os.path.join(dirs, "normal_val.feather")
    val_data.reset_index(drop=True).to_feather(filename)


def create_data(variant):
    #create_mae_data(variant)
    #create_timing_data(variant)
    #standard_data(variant)
    build_data(variant)

def load_micro(method, window, horizon, time_format='%Y-%m-%d'):
    pdb.set_trace()
    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'mixture')
    train_data = pd.read_feather(os.path.join(dirs, "normal_train.feather"))
    val_data = pd.read_feather(os.path.join(dirs, "normal_val.feather"))

    long_temporal_features = train_data.filter(
        regex="^long").columns.to_list()

    medium_temporal_features = train_data.filter(
        regex="^medium").columns.to_list()

    short_temporal_features = train_data.filter(
        regex="^short").columns.to_list()

    mae_temporal_features = train_data.filter(
        regex="^mae").columns.to_list()
    long_temporal_features = sorted(long_temporal_features, key=lambda s: int(re.search(r'\d+', s).group()))
    medium_temporal_features = sorted(medium_temporal_features, key=lambda s: int(re.search(r'\d+', s).group()))
    short_temporal_features = sorted(short_temporal_features, key=lambda s: int(re.search(r'\d+', s).group()))
    mae_temporal_features = sorted(mae_temporal_features, key=lambda s: int(re.search(r'\d+', s).group()))

    features = mae_temporal_features + short_temporal_features + medium_temporal_features + long_temporal_features
    train_data = train_data[['trade_date','code'] + features + ['nxt1_ret_{0}h'.format(horizon)]]
    val_data = val_data[['trade_date','code'] + features + ['nxt1_ret_{0}h'.format(horizon)]]
    train_data.rename(columns={'trade_date':'trade_time','nxt1_ret_{0}h'.format(horizon):'nxt1_ret'}, inplace=True)
    val_data.rename(columns={'trade_date':'trade_time','nxt1_ret_{0}h'.format(horizon):'nxt1_ret'}, inplace=True)
    pdb.set_trace()
    codes = train_data.code.unique().tolist()
    
    train_dataset = CogniDataSet8.generate(train_data,
                                           codes=codes,
                                           features=features,
                                           window=window,
                                           target=['nxt1_ret'],
                                           time_name='trade_time',
                                           time_format=time_format)
    

    val_dataset = CogniDataSet8.generate(val_data,
                                         codes=codes,
                                         features=features,
                                         window=window,
                                         target=['nxt1_ret'],
                                         time_name='trade_time',
                                         time_format=time_format)
    return train_dataset, val_dataset, mae_temporal_features, long_temporal_features, medium_temporal_features, short_temporal_features


def load_misro(method, window, horizon, time_format='%Y-%m-%d'):
    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'mixture')
    val_data = pd.read_feather(os.path.join(dirs, "normal_val.feather"))

    long_temporal_features = val_data.filter(
        regex="^long").columns.to_list()

    medium_temporal_features = val_data.filter(
        regex="^medium").columns.to_list()

    short_temporal_features = val_data.filter(
        regex="^short").columns.to_list()

    mae_temporal_features = val_data.filter(
        regex="^mae").columns.to_list()
    long_temporal_features = sorted(long_temporal_features, key=lambda s: int(re.search(r'\d+', s).group()))
    medium_temporal_features = sorted(medium_temporal_features, key=lambda s: int(re.search(r'\d+', s).group()))
    short_temporal_features = sorted(short_temporal_features, key=lambda s: int(re.search(r'\d+', s).group()))
    mae_temporal_features = sorted(mae_temporal_features, key=lambda s: int(re.search(r'\d+', s).group()))

    features = mae_temporal_features + short_temporal_features + medium_temporal_features + long_temporal_features
    val_data = val_data[['trade_date','code'] + features + ['nxt1_ret_{0}h'.format(horizon)]]
    val_data.rename(columns={'trade_date':'trade_time','nxt1_ret_{0}h'.format(horizon):'nxt1_ret'}, inplace=True)
    pdb.set_trace()
    codes = val_data.code.unique().tolist()
    

    window_val_data = CogniDataSet8.create_windows(
        val_data.set_index(['trade_time','code']),
                                         features=features,
                                         window=window,
                                         time_name='trade_time')
    return window_val_data, mae_temporal_features, long_temporal_features, medium_temporal_features, short_temporal_features


    
def train(variant):
    batch_size = 32
    train_dataset, val_dataset, mae_features, long_temporal_features, medium_temporal_features, short_temporal_features = load_micro(method=variant['method'], window=variant['window'], horizon=variant['horizon'])

    pdb.set_trace()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    pawn = Pawn0004(id="merak_{0}_{1}h".format(variant['method'],
                                                   variant['horizon']),
                    features=train_dataset.features,
                    ticker_count=len(train_dataset.code.unique()),
                    window=variant['window'],
                    relational_dim=len(mae_features),
                    long_dim=len(long_temporal_features),
                    short_dim=len(short_temporal_features),
                    medium_dim=len(medium_temporal_features))
    max_episode = 110
    pawn._zeatures_atten2.train_model(train_loader=train_loader,
                                      val_loader=val_loader,
                                      is_state_dict=True,
                                      model_dir=g_train_path,
                                      tb_dir=g_tensorboard_path,
                                      push_dir=g_push_path,
                                      epochs=max_episode)


def predict(variant):
    window_val_data, mae_features, long_temporal_features, medium_temporal_features, short_temporal_features = load_misro(
        method=variant['method'], window=variant['window'], horizon=variant['horizon'])
    pdb.set_trace()
    pawn = Pawn0004(id="merak_{0}_{1}h".format(variant['method'],
                                                   variant['horizon']),
                    features=window_val_data['features'],
                    ticker_count=len(window_val_data['code'].unique()),
                    window=variant['window'],
                    relational_dim=len(mae_features),
                    long_dim=len(long_temporal_features),
                    short_dim=len(short_temporal_features),
                    medium_dim=len(medium_temporal_features),
                    is_load=True)
    
    factors_name = ["factor_{0}".format(i) for i in range(0, 1)]

    hidden_name = ["hidden_{0}".format(i) for i in range(0, 256)]

    res = []
    for k, v in window_val_data['data'].groupby('trade_time'):
        print(k)
        output = pawn.predict(
            to_device(torch.from_numpy(v.values[np.newaxis, :, :])), False)

        factors = pd.DataFrame(np.squeeze(output[-1].detach().cpu(), axis=0),
                               index=v.index,
                               columns=factors_name)

        hidden = pd.DataFrame(np.squeeze(output[0].detach().cpu(), axis=0),
                              index=v.index,
                              columns=hidden_name)

        res.append(pd.concat([factors, hidden], axis=1))

    factors_data = pd.concat(res)
    pdb.set_trace()
    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'mixture', pawn.kid)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs, 'pawn0004.feather')
    factors_data.reset_index().to_feather(filename)



def metrics(variant):
    base_path = 'pawn0004_2p_2s_ranking_1h_merak_hs300_1h'
    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'mixture', base_path)
    filename = os.path.join(dirs, 'pawn0004.feather')
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
    base_path = 'pawn0004_2p_2s_ranking_1h_merak_hs300_1h'
    file_path = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], 'mixture', base_path, 'metrics')
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

    #create_data(vars(args))
    #train(vars(args))
    predict(vars(args))
    #metrics(vars(args))
    #polymer(vars(args))