import os, copy
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()


from kdutils.tactix import Tactix
from kdutils.macro2 import *
from lib.uvx import * 

from lib.rl014.train import train_model
from lib.rl014.predict import predict_test_set
from lib.rl014.analysis import create_evaluate
# from lib.rl011.analysis import analyze_run


def _sanitize_frame(df: pd.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    bad_mask = ~np.isfinite(df[cols].to_numpy(dtype=np.float64))
    bad_count = int(bad_mask.sum())
    if bad_count > 0:
        print(f"[WARN] 数据中发现 {bad_count} 个 NaN/Inf，已填充为 0.0")
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def load_data02(method, instruments, task_id, period, features, regime, ret_name, category):
    def add_canary(df, rho=0.6, seed=42):
        rng = np.random.default_rng(seed)
        y = df['nxt1_ret'].astype(float).to_numpy()

        mu = np.nanmean(y)
        sd = np.nanstd(y) + 1e-8

        zy = (y - mu) / sd
        e = rng.normal(size=len(df))
        x = rho * zy + np.sqrt(1 - rho**2) * e 
        df[f"CANARY_GOOD_{int(rho*100):02d}"] = x
        return df


    base_dirs = os.path.join(base_path, method, instruments, 'temp',
                             'model', str(task_id), str(period),
                               'rl', 'data')
    if category == 'train':
        data = pd.read_feather(os.path.join(base_dirs, "train_data.feather"))
    elif category == 'val':
        data = pd.read_feather(os.path.join(base_dirs, "val_data.feather"))
    elif category == 'test':
        data = pd.read_feather(os.path.join(base_dirs, "test_data.feather"))

    data.rename(columns={ret_name: "nxt1_ret"}, inplace=True)
    data = add_canary(data, 0.2)
    data = data[['trade_time', 'code', 'nxt1_ret'] + features + regime]
    data = data.sort_values('trade_time').reset_index(drop=True)
    
    
    data = _sanitize_frame(data, ['nxt1_ret'] + features + regime)
    if data['code'].nunique() != 1:
        raise ValueError(f"test_data 不是单标的，检测到 {data['code'].nunique()} 个 code")
    return data



def load_data12(method, instruments, task_id, period, features, regime, ret_name):
    def add_canary(df, rho=0.6, seed=42):
        rng = np.random.default_rng(seed)
        y = df['nxt1_ret'].astype(float).to_numpy()

        mu = np.nanmean(y)
        sd = np.nanstd(y) + 1e-8

        zy = (y - mu) / sd
        e = rng.normal(size=len(df))
        x = rho * zy + np.sqrt(1 - rho**2) * e 
        df[f"CANARY_GOOD_{int(rho*100):02d}"] = x
        return df

    base_dirs = os.path.join(base_path, method, instruments, 'temp',
                             'model', str(task_id), str(period),
                               'rl', 'data')
    train_data = pd.read_feather(os.path.join(base_dirs,
                                              "train_data.feather"))
    
    val_data = pd.read_feather(os.path.join(base_dirs,
                                              "val_data.feather"))
    
    train_data.rename(columns={ret_name: "nxt1_ret"},
                      inplace=True)

    val_data.rename(columns={ret_name: "nxt1_ret"},
                    inplace=True)
    
    train_data = add_canary(train_data, 0.2)
    val_data = add_canary(val_data, 0.2)
    
    train_data = train_data[['trade_time','code', 'nxt1_ret'] + features + regime]
    val_data = val_data[['trade_time','code', 'nxt1_ret'] + features + regime]
    
    
    train_data = train_data.sort_values('trade_time').reset_index(drop=True)
    val_data = val_data.sort_values('trade_time').reset_index(drop=True)
    train_data = _sanitize_frame(train_data, ['nxt1_ret'] + features + regime)
    val_data = _sanitize_frame(val_data, ['nxt1_ret'] + features + regime)
    if train_data['code'].nunique() != 1:
        raise ValueError(f"train_data 不是单标的，检测到 {train_data['code'].nunique()} 个 code")
    if val_data['code'].nunique() != 1:
        raise ValueError(f"val_data 不是单标的，检测到 {val_data['code'].nunique()} 个 code")
    return train_data, val_data

    
def load_data1(method, instruments, task_id, period, features, regime, ret_name):
    base_dirs = os.path.join(base_path, method, instruments, 'temp',
                             'model', str(task_id), str(period),
                               'rl', 'data')
    train_data = pd.read_feather(os.path.join(base_dirs,
                                              "train_data.feather"))
    
    val_data = pd.read_feather(os.path.join(base_dirs,
                                              "val_data.feather"))
    
    train_data.rename(columns={ret_name: "nxt1_ret"},
                      inplace=True)

    val_data.rename(columns={ret_name: "nxt1_ret"},
                    inplace=True)
    
    train_data = train_data[['trade_time','code', 'nxt1_ret'] + features + regime]
    val_data = val_data[['trade_time','code', 'nxt1_ret'] + features + regime]
    
    train_data = train_data.sort_values('trade_time').reset_index(drop=True)
    val_data = val_data.sort_values('trade_time').reset_index(drop=True)
    train_data = _sanitize_frame(train_data, ['nxt1_ret'] + features + regime)
    val_data = _sanitize_frame(val_data, ['nxt1_ret'] + features + regime)
    if train_data['code'].nunique() != 1:
        raise ValueError(f"train_data 不是单标的，检测到 {train_data['code'].nunique()} 个 code")
    if val_data['code'].nunique() != 1:
        raise ValueError(f"val_data 不是单标的，检测到 {val_data['code'].nunique()} 个 code")
    return train_data, val_data


def load_data2(method, instruments, task_id, period, features, regime, ret_name):
    base_dirs = os.path.join(base_path, method, instruments, 'temp',
                             'model', str(task_id), str(period),
                               'rl', 'data')
    test_data = pd.read_feather(os.path.join(base_dirs, "test_data.feather"))

    test_data.rename(columns={ret_name: "nxt1_ret"}, inplace=True)
    test_data = test_data[['trade_time', 'code', 'nxt1_ret'] + features + regime]
    test_data = test_data.sort_values('trade_time').reset_index(drop=True)
    test_data = _sanitize_frame(test_data, ['nxt1_ret'] + features + regime)
    if test_data['code'].nunique() != 1:
        raise ValueError(f"test_data 不是单标的，检测到 {test_data['code'].nunique()} 个 code")
    return test_data


def load_data0(method, instruments, task_id, period, features, regime, ret_name, category):
    base_dirs = os.path.join(base_path, method, instruments, 'temp',
                             'model', str(task_id), str(period),
                               'rl', 'data')
    if category == 'train':
        data = pd.read_feather(os.path.join(base_dirs, "train_data.feather"))
    elif category == 'val':
        data = pd.read_feather(os.path.join(base_dirs, "val_data.feather"))
    elif category == 'test':
        data = pd.read_feather(os.path.join(base_dirs, "test_data.feather"))

    data.rename(columns={ret_name: "nxt1_ret"}, inplace=True)
    data = data[['trade_time', 'code', 'nxt1_ret'] + features + regime]
    data = data.sort_values('trade_time').reset_index(drop=True)
    data = _sanitize_frame(data, ['nxt1_ret'] + features + regime)
    if data['code'].nunique() != 1:
        raise ValueError(f"test_data 不是单标的，检测到 {data['code'].nunique()} 个 code")
    return data



def train(method, instruments, task_id, period, env_id, trade_id, model_id, train_id, feature_id, regime_id):
    file_dirs = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl')
    
    env_params, trade_params, model_params, train_params, selected_features, min_regime, daily_regime = load_rl_params(
        file_dirs=file_dirs, 
        trade_id=trade_id, model_id=model_id, feature_id=feature_id,
        env_id=env_id, train_id=train_id, regime_id=regime_id)

    # selected_features += ['CANARY_GOOD_20']
    
    total_params = copy.deepcopy(trade_params)
    total_params.update(env_params)
    total_params.update(model_params)
    total_params.update(train_params)
    total_params.update({'selected_features':selected_features})
    total_params.update({'min_regime':min_regime})
    total_params.update({'daily_regime':daily_regime})
    
    name = Params.create_tag(total_params)
    
    output_dir = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl', 'result',str(name))
    tensorboard_dir = os.path.join(tensorboard_path, instruments, str(task_id), str(name))
    
    # tensorboard_dir = os.path.join(base_path, method, instruments, 'temp',
    #                            'model', str(task_id), str(period),
    #                            'rl', "tensorboard", str(name))
    
    os.makedirs(output_dir, exist_ok=True)
    logger.configure(log_file=os.path.join(output_dir,f"model.log"))
    
    
    train_data, val_data = load_data1(method=method, instruments=instruments, period=period, 
                                      task_id=task_id, ret_name=trade_params['ret_name'],
                                      features=selected_features, regime=min_regime)
    
    env_config = {
        'holding_period': int(env_params['holding_period']),
        'reward_scale': float(env_params['reward_scale']),
        'action_change_penalty': float(env_params['action_change_penalty']),
        'action_change_deadzone': float(env_params['action_change_deadzone']),
        'min_open_signal_abs': float(env_params['min_open_signal_abs']),
        'max_episode_steps': float(env_params['max_episode_steps']),
        'train_scheme': env_params['train_scheme'],
        'softmax_temperature':env_params['softmax_temperature'],
        'seed': env_params['seed'],
        'target_mode': env_params['target_mode'],
        'target_cost_rate': float(env_params.get('target_cost_rate', COST_MAPPING[INSTRUMENTS_CODES[instruments]])
        ),
        'target_cost_mult': float(env_params['target_cost_mult']),
    }
    
    sac_config = {
        'learning_rate': model_params['learning_rate'],
        'buffer_size': model_params['buffer_size'],  # 示例使用较小缓冲区
        'learning_starts': model_params['learning_starts'],
        'batch_size':  model_params['batch_size'],
        'tau':  model_params['tau'],
        'gamma':  model_params['gamma'],
        'train_freq':  model_params['train_freq'],
        'gradient_steps':  model_params['gradient_steps'],
        'ent_coef':  model_params['ent_coef'],
        'target_update_interval':  model_params['target_update_interval'],
        'policy_kwargs': {
            'net_arch': {
                'pi': model_params['policy_kwargs']['net_arch']['pi'],
                'qf': model_params['policy_kwargs']['net_arch']['qf']
            }
        }
    }
    
    signal_config = {
        # RL011：action 默认直接作为连续 ER 使用
        # 若要离散化可在参数里提供 discrete_mode/discrete_threshold
        'min_open_signal_abs': float(env_params['min_open_signal_abs'])
    }
    
    eval_n_episodes = int(train_params['eval_n_episodes'])
    early_stop_patience_evals = int(train_params['early_stop_patience_evals'])
    early_stop_min_evals = int(train_params['early_stop_min_evals'])
    early_stop_min_delta = float(train_params['early_stop_min_delta'])
    early_stop_start_timesteps =  int(float(train_params['early_stop_start_timesteps']))#int(model_params['learning_starts']) + int(train_params['eval_freq'])
    
    
    logger.info(f"  训练集: {len(train_data)} 行")
    logger.info(f"  校验集: {len(val_data)} 行")
    
    logger.info(f" env_config: {env_config}")
    logger.info(f" sac_config: {sac_config}")
    logger.info(f" signal_config: {signal_config}")
    logger.info(f" train_params: {train_params}")
    logger.info(f" env_params: {env_params}")
    logger.info(f" trade_params: {trade_params}")
    logger.info(f" model_params: {model_params}")
    logger.info(f" selected_features: {selected_features}")
    logger.info(f" min_regime: {min_regime}")
    logger.info(f" daily_regime: {daily_regime}")
    
    model, training_info = train_model(
            train_df=train_data,
            val_df=val_data,
            features=selected_features + min_regime,
            env_config=env_config,
            sac_config=sac_config,
            signal_config=signal_config,
            output_dir=output_dir,
            tensorboard_dir=tensorboard_dir,
            total_timesteps=train_params['total_timesteps'],
            eval_freq=train_params['eval_freq'],
            save_freq=train_params['save_freq'],
            eval_n_episodes=eval_n_episodes,
            early_stop_patience_evals=early_stop_patience_evals,
            early_stop_min_evals=early_stop_min_evals,
            early_stop_min_delta=early_stop_min_delta,
            early_stop_start_timesteps=early_stop_start_timesteps,
            verbose=1
        )


def predict(method, instruments, task_id, period, env_id, trade_id, 
            model_id, train_id, feature_id, regime_id):
    file_dirs = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl')

    env_params, trade_params, model_params, train_params, selected_features, min_regime, daily_regime = load_rl_params(
        file_dirs=file_dirs,
        trade_id=trade_id, model_id=model_id, feature_id=feature_id,
        env_id=env_id, train_id=train_id, regime_id=regime_id
    )
    
    total_params = copy.deepcopy(trade_params)
    total_params.update(env_params)
    total_params.update(model_params)
    total_params.update(train_params)
    total_params.update({'selected_features': selected_features})
    total_params.update({'min_regime': min_regime})
    total_params.update({'daily_regime': daily_regime})
    name = Params.create_tag(total_params)

    # test_data = load_data2(
    #     method=method, instruments=instruments, period=period,
    #     task_id=task_id, ret_name=trade_params['ret_name'],
    #     features=selected_features, regime=min_regime
    # )
    

    output_dir = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl', 'result', str(name))
    best_model_path = os.path.join(output_dir, "models", "best_model", "best_model")
    config_path = os.path.join(output_dir, "config.json")

    
    
    for category in ['val','test']:
        data = load_data0(
            method=method, instruments=instruments, period=period,
            task_id=task_id, ret_name=trade_params['ret_name'],
            features=selected_features, regime=min_regime, 
            category=category
        )
    
        filename = os.path.join(output_dir, "metrics", "{0}_results.csv".format(category))
        # image_path = os.path.join(output_dir, "metrics", "{0}_results.png".format(category))
        predict_test_set(
            model_path=best_model_path,
            config_path=config_path,
            test_df=data,
            output_path=filename,
            deterministic=True
        )
        # df1 = pd.read_csv(filename)
        # create_evaluate(df=df1, factor_name='net_er_out', return_name='future_ret_h', title_prefix=category, image_path=image_path)

def evaluate(method, instruments, task_id, period, env_id, trade_id, 
            model_id, train_id, feature_id, regime_id):
    file_dirs = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl')

    env_params, trade_params, model_params, train_params, selected_features, min_regime, daily_regime = load_rl_params(
        file_dirs=file_dirs,
        trade_id=trade_id, model_id=model_id, feature_id=feature_id,
        env_id=env_id, train_id=train_id, regime_id=regime_id
    )
    # selected_features += ['CANARY_GOOD_20']
    total_params = copy.deepcopy(trade_params)
    total_params.update(env_params)
    total_params.update(model_params)
    total_params.update(train_params)
    total_params.update({'selected_features': selected_features})
    total_params.update({'min_regime': min_regime})
    total_params.update({'daily_regime': daily_regime})
    name = Params.create_tag(total_params)
    
    output_dir = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl', 'result', str(name))
    pnl_method = 'points_norm'
    for category in ['val','test']:
        filename = os.path.join(output_dir, "metrics", "{0}_results.csv".format(category))
        image_path = os.path.join(output_dir, "metrics", "{0}_results_{1}.png".format(category, pnl_method))
        print(filename)
        print(image_path)
        df1 = pd.read_csv(filename)
        # abs_er = pd.read_csv(filename)["er_value"].abs() 
        # base_open = (abs_er > 0).mean()  # baseline 开仓率（等价 er!=0）

        # for thr in [0.003, 0.005, 0.01, 0.02]:
        #     open_thr = ((abs_er >= thr) & (abs_er > 0)).mean()
        #     print(thr, open_thr, "drop_vs_base=", base_open - open_thr)
        
        
        pdb.set_trace()
        create_evaluate(df=df1, factor_name='net_er_out', return_name='future_ret_h',
                        title_prefix=category, image_path=image_path,pnl_method=pnl_method,
                        cost_rate=COST_MAPPING[INSTRUMENTS_CODES[instruments]])
    
if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == "train":
        train(method=variant.method, 
          instruments=variant.instruments, 
          task_id=variant.task_id, 
          period=variant.period, 
          env_id=variant.env_id, 
          trade_id=variant.trade_id, 
          model_id=variant.model_id, 
          train_id=variant.train_id,
          feature_id=variant.feature_id,
          regime_id=variant.regime_id)
    elif variant.form == "predict":
        predict(method=variant.method, 
          instruments=variant.instruments, 
          task_id=variant.task_id, 
          period=variant.period, 
          env_id=variant.env_id, 
          trade_id=variant.trade_id, 
          model_id=variant.model_id, 
          train_id=variant.train_id,
          feature_id=variant.feature_id,
          regime_id=variant.regime_id)
    elif variant.form == "eval":
        evaluate(method=variant.method, 
          instruments=variant.instruments, 
          task_id=variant.task_id, 
          period=variant.period, 
          env_id=variant.env_id, 
          trade_id=variant.trade_id, 
          model_id=variant.model_id, 
          train_id=variant.train_id,
          feature_id=variant.feature_id,
          regime_id=variant.regime_id)
    #df1 = pd.read_csv("./records/cicso1/ims/temp/model/200037/15/rl/result/1077258471563928/metrics/train_results.csv")
    #create_evaluate(df=df1, factor_name='net_er_out', return_name='future_ret_h')
