import os, pdb
import torch
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from ultron.optimize.wisem import *
from kdutils.macro2 import *
from lib.lsx001 import fetch_times
from lib.svx001 import scale_factors

from kichaos.datasets import CogniDataSet1 as CogniDataSet
from kichaos.agent.envoy.envoy0010 import Envoy0010


def load_data(method, instruments, task_id, period):
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    filename = os.path.join(dirs, "final_data.feather")
    final_data = pd.read_feather(filename)  #.set_index(['trade_time', 'code'])

    features = [
        f for f in final_data.columns
        if f not in ['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)]
    ]
    new_features = ["feature_{0}".format(f) for f in range(0, len(features))]
    final_data.columns = ['trade_time', 'code'
                          ] + new_features + ['nxt1_ret_{0}h'.format(period)]

    return final_data.set_index(['trade_time', 'code'])


def standard_features(prepare_features, method, win):
    features = prepare_features.columns
    predict_data = prepare_features.copy().dropna()
    for f in features:
        scale_factors(predict_data=predict_data,
                      method=method,
                      win=win,
                      factor_name=f)
        prepare_features[f] = predict_data['transformed']
    return prepare_features


def create_data1(method,
                 instruments,
                 task_id,
                 period,
                 sethod='roll_zscore',
                 win=240):
    pdb.set_trace()
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    prepare_pd = load_data(method=method,
                           instruments=instruments,
                           task_id=task_id,
                           period=period)
    standard_features(prepare_features=prepare_pd, method=sethod, win=win)
    ### 切割数据
    train_data = prepare_pd.loc[
        time_array['train_time'][0]:time_array['train_time'][1]].reset_index(
        ).sort_values(['trade_time', 'code'])
    val_data = prepare_pd.loc[time_array['val_time'][0]:time_array['val_time']
                              [1]].reset_index().sort_values(
                                  ['trade_time', 'code'])

    train_data.rename(columns={'nxt1_ret_{0}h'.format(period): 'nxt1_ret'},
                      inplace=True)
    val_data.rename(columns={'nxt1_ret_{0}h'.format(period): 'nxt1_ret'},
                    inplace=True)

    ### 新增 对y 标准化
    # 1. 从训练集中计算 y 的均值和标准差
    y_train_mean = train_data['nxt1_ret'].mean()
    y_train_std = train_data['nxt1_ret'].std()
    
    print(f"--- Standardizing y (nxt1_ret) ---")
    print(f"Train set mean: {y_train_mean}, std: {y_train_std}")

    # 2. 使用训练集的统计量来标准化训练集和验证集
    train_data['nxt1_ret'] = (train_data['nxt1_ret'] - y_train_mean) / y_train_std
    val_data['nxt1_ret'] = (val_data['nxt1_ret'] - y_train_mean) / y_train_std
    

    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code', 'nxt1_ret']
    ]
    pdb.set_trace()

    train_dataset = CogniDataSet.generate(data=train_data.dropna(),
                                          features=features,
                                          window=1,
                                          time_name='trade_time',
                                          target=['nxt1_ret'])

    val_dataset = CogniDataSet.generate(data=val_data.dropna(),
                                        features=features,
                                        window=1,
                                        time_name='trade_time',
                                        target=['nxt1_ret'])
    return train_dataset, val_dataset, train_data, val_data


def create_data2(method,
                 instruments,
                 task_id,
                 period,
                 sethod='roll_zscore',
                 win=240):
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    prepare_pd = load_data(method=method,
                           instruments=instruments,
                           task_id=task_id,
                           period=period)
    standard_features(prepare_features=prepare_pd, method=sethod, win=win)
    ### 切割数据
    train_data = prepare_pd.loc[
        time_array['train_time'][0]:time_array['train_time'][1]].reset_index(
        ).sort_values(['trade_time', 'code'])

    val_data = prepare_pd.loc[time_array['val_time'][0]:time_array['val_time']
                              [1]].reset_index().sort_values(
                                  ['trade_time', 'code'])

    test_data = prepare_pd.loc[
        time_array['test_time'][0]:time_array['test_time'][1]].reset_index(
        ).sort_values(['trade_time', 'code'])

    train_data.rename(columns={'nxt1_ret_{0}h'.format(period): 'nxt1_ret'},
                      inplace=True)
    val_data.rename(columns={'nxt1_ret_{0}h'.format(period): 'nxt1_ret'},
                    inplace=True)
    test_data.rename(columns={'nxt1_ret_{0}h'.format(period): 'nxt1_ret'},
                     inplace=True)

    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code', 'nxt1_ret']
    ]

    pdb.set_trace()
    window_test_data = CogniDataSet.create_windows(data=test_data.set_index(
        ['trade_time', 'code']),
                                                   features=features,
                                                   window=1,
                                                   time_name='trade_time')
    return window_test_data, test_data


def train_model(method, task_id, instruments, period):
    batch_size = 64
    max_episode = 20
    train_dataset, val_dataset, train_data, val_data = create_data1(
        method=method, instruments=instruments, task_id=task_id, period=period)
    pdb.set_trace()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    envoy = Envoy0010(id="mizar_{0}_{1}_{2}".format(method, task_id,
                                                    instruments),
                      features=train_dataset.features,
                      ticker_count=len(train_dataset.code.unique()),
                      window=1,
                      is_debug=False)
    envoy._normal_temporal_convolution.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        is_state_dict=True,
        model_dir=envoy.train_path,
        tb_dir=envoy.tensorboard_path,
        push_dir=envoy.push_path,
        epochs=max_episode)


def predict_model(method, task_id, instruments, period):
    window_data, total_data = create_data2(method=method,
                                           task_id=task_id,
                                           instruments=instruments,
                                           period=period)
    pdb.set_trace()
    envoy = Envoy0010(id="mizar_{0}_{1}_{2}".format(method, task_id,
                                                    instruments),
                      features=window_data['features'],
                      ticker_count=len(window_data['code'].unique()),
                      window=1,
                      is_load=True,
                      is_debug=False)
    envoy._normal_temporal_convolution.eval()
    temporal_feature = torch.from_numpy(
        window_data['data'][window_data['wfeatures']].values).to(
            envoy.device).reshape(-1,
                                  len(window_data['features']) * 1,
                                  len(window_data['code'].unique())).float()
    with torch.no_grad():
        output_tensor = envoy._normal_temporal_convolution.predict(
            temporal_feature)
        outputs = output_tensor.detach().cpu().numpy()
    outputs = pd.DataFrame(outputs,
                           index=window_data['data'].index,
                           columns=['predict'])

    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    pdb.set_trace()
    outputs = outputs.reset_index().merge(
        total_data[['trade_time', 'code', 'nxt1_ret']],
        on=['trade_time', 'code'])
    outputs = outputs.rename(
        columns={'nxt1_ret': 'nxt1_ret_{}h'.format(period)})
    pdb.set_trace()
    outputs.reset_index().to_feather(
        os.path.join(dirs, "tcn_predict_data.feather"))
    '''
    res = []
    for k, v in window_data['data'].groupby('trade_time'):
        print(k)
        pdb.set_trace()
        array = window_data['data'].loc[k][window_data['wfeatures']].values
        temporal_feature = torch.from_numpy(array).to(envoy.device).reshape(
            -1,
            len(window_data['features']) * 1,
            len(window_data['code'].unique())).float()
        with torch.no_grad():
            output_tensor = envoy._normal_temporal_convolution.predict(temporal_feature)
            outputs = output_tensor.detach().cpu().numpy()
            res.append(outputs)
    '''


train_model(method='bicso0', task_id='113001', instruments='rbb', period=5)
#predict_model(method='bicso0', task_id='113001', instruments='rbb', period=5)
