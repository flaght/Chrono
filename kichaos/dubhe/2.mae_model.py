import sys, os, pdb, argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.append('../../kichaos')

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from ultron.optimize.wisem import *
from kdutils.macro import base_path
from kdutils.data import *
from kdutils.metrics import metrics as kdmetrics

from kichaos.datasets import CogniDataSet6
from kichaos.agent import Liaison0004


def load_misro(method, window, universe, horzion=1, time_format='%Y-%m-%d'):
    filename = os.path.join(base_path, universe, str(horzion),
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

    pos = int(len(dates) * 0.7)

    _ = total_data[total_data['trade_date'].isin(dates[:pos])]
    val_data = total_data[total_data['trade_date'].isin(dates[pos - window +
                                                              1:])]

    features = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code', 'dummy', 'nxt1_ret'] +
        nxt1_columns
    ]

    window_val_data = CogniDataSet6.create_windows(data=val_data.set_index(
        ['trade_date', 'code']),
                                                   features=features,
                                                   target=nxt1_columns,
                                                   window=window,
                                                   time_name='trade_date')
    return window_val_data


def load_micro(method,
               window=3,
               universe=None,
               horzion=1,
               time_format='%Y-%m-%d'):
    ##
    pdb.set_trace()
    filename = os.path.join(base_path, universe, str(horzion),
                            "{0}_model_normal.feather".format(method))
    total_data = pd.read_feather(filename)
    total_data = total_data.sort_values(['trade_date', 'code'])

    nxt1_columns = total_data.filter(regex="^nxt1_").columns.to_list()
    columns = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code'] + nxt1_columns
    ]
    pdb.set_trace()
    #columns = columns[0:15]
    total_data = total_data[['trade_date', 'code'] + columns +
                            nxt1_columns].sort_values(['trade_date', 'code'])
    dates = total_data['trade_date'].dt.strftime(time_format).unique().tolist()

    pos = int(len(dates) * 0.7)

    train_data = total_data[total_data['trade_date'].isin(dates[:pos])]
    val_data = total_data[total_data['trade_date'].isin(dates[pos:])]
    pdb.set_trace()
    codes = total_data.code.unique().tolist()

    features = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code', 'chg_pct'] + nxt1_columns
    ]
    train_dataset = CogniDataSet6.generate(train_data,
                                           codes=codes,
                                           features=features,
                                           window=window,
                                           target=nxt1_columns,
                                           time_name='trade_date',
                                           time_format=time_format)

    val_dataset = CogniDataSet6.generate(val_data,
                                         codes=codes,
                                         features=features,
                                         window=window,
                                         target=nxt1_columns,
                                         time_name='trade_date',
                                         time_format=time_format)
    return train_dataset, val_dataset


def train(variant):
    batch_size = 32
    train_dataset, val_dataset = load_micro(method=variant['method'],
                                            window=variant['window'],
                                            universe=variant['universe'],
                                            horzion=variant['horzion'])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    pdb.set_trace()
    envoy = Liaison0004(id="dubhe_{0}_{1}".format(variant['method'],
                                                  variant['universe']),
                        features=train_dataset.features,
                        targets=train_dataset.target,
                        ticker_count=len(train_dataset.code.unique()),
                        window=variant['window'])
    max_episode = 200
    envoy._planar_hybrid_transformer.train_model(train_loader=train_loader,
                                                 val_loader=val_loader,
                                                 is_state_dict=True,
                                                 model_dir=envoy.train_path,
                                                 tb_dir=envoy.tensorboard_path,
                                                 push_dir=envoy.push_path,
                                                 epochs=max_episode)


def predict(variant):
    val_data = load_misro(method=variant['method'],
                          window=variant['window'],
                          universe=variant['universe'])

    pdb.set_trace()
    envoy = Liaison0004(id="dubhe_{0}_{1}".format(variant['method'],
                                                  variant['universe']),
                        features=val_data['features'],
                        targets=val_data['targets'],
                        ticker_count=len(val_data['code'].unique()),
                        window=variant['window'],
                        is_load=True)
    factors_name = [
        "factor_{0}".format(i) for i in range(0, len(val_data['targets']))
    ]
    hidden_name = ["hidden_{0}".format(i) for i in range(0, envoy.dimension)]
    res1_out = []
    res2_out = []

    for k, v in val_data['data'].groupby('trade_date'):
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

    dirs = os.path.join(base_path, variant['universe'],
                        str(variant['horzion']), envoy.kid)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs, "liaison0004_factors.feather")
    factors_data.reset_index().to_feather(filename)

    filename = os.path.join(dirs, "liaison0004_hidden.feather")
    pdb.set_trace()
    hidden_data.reset_index().to_feather(filename)


def metrics(variant):
    pdb.set_trace()
    model_path = 'liaison0004_8p_3s_ranking_state_dubhe_sicro_hs300'
    filename = os.path.join(base_path, variant['universe'],
                            str(variant['horzion']), model_path,
                            "liaison0004_factors.feather")
    factors_data = pd.read_feather(filename)
    factors_data = factors_data.drop(['time_stat'], axis=1)

    filename = os.path.join(base_path, variant['universe'],
                            str(variant['horzion']), model_path,
                            "liaison0004_hidden.feather")

    hidden_data = pd.read_feather(filename).drop(['time_stat'], axis=1)

    factors_data = factors_data.merge(hidden_data,
                                      on=['trade_date', 'code'],
                                      how='left')
    begin_date = factors_data['trade_date'].min().strftime('%Y-%m-%d')
    end_date = factors_data['trade_date'].max().strftime('%Y-%m-%d')
    remain_data = fetch_f1r_oo(begin_date, end_date, variant['universe'])
    dummy_fst = remain_data['dummy_test_f1r_open'] * remain_data[
        variant['universe']]
    yields_data = remain_data['ret_f1r_oo'] * remain_data[variant['universe']]
    factors_data = factors_data.set_index(['trade_date', 'code'])
    factor_columns = factors_data.columns
    res = []
    dirs = os.path.join(base_path, variant['universe'],
                        str(variant['horzion']), 'metrics', model_path,
                        'liaison0004_{0}'.format(variant['name']))
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
    print(dirs)
    print('done')


def polymer(variant):
    model_path = 'liaison0004_8p_3s_ranking_state_dubhe_sicro_hs300'
    file_path = os.path.join(base_path, variant['universe'],
                             str(variant['horzion']), 'metrics', model_path,
                             'liaison0004_{0}'.format(variant['name']))
    print(file_path)
    pdb.set_trace()
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
    pdb.set_trace()
    print(metrics1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='sicro')
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--name', type=str, default='factors')
    parser.add_argument('--horzion', type=int, default=1)
    parser.add_argument('--universe', type=str, default='hs300')

    args = parser.parse_args()

    #train(vars(args))
    #predict(vars(args))
    metrics(vars(args))
    polymer(vars(args))
