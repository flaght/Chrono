import os, copy, pdb
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()

from kdutils.tactix import Tactix
from kdutils.macro2 import *
from lib.uvx import *
from lib.rl015.train import train_model

PAIRE_TASK = {"113001": ("hcb", "134001")}


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


def _merge_assets(left_data, right_data):
    data = pd.concat([left_data, right_data], ignore_index=True)
    data['trade_time'] = pd.to_datetime(data['trade_time'], errors='raise')
    if data['trade_time'].isna().any() or data.duplicated(
        ['code', 'trade_time']).any():
        raise ValueError("品种时间键缺失或重复")
    return data.sort_values(['code', 'trade_time']).reset_index(drop=True)


def load_data1(method, instruments, task_id, period, features, regime,
               ret_name):
    base_dirs = os.path.join(base_path, method, instruments, 'temp', 'model',
                             str(task_id), str(period), 'rl', 'data')
    train_data = pd.read_feather(os.path.join(base_dirs, "train_data.feather"))

    val_data = pd.read_feather(os.path.join(base_dirs, "val_data.feather"))

    train_data.rename(columns={ret_name: "nxt1_ret"}, inplace=True)

    val_data.rename(columns={ret_name: "nxt1_ret"}, inplace=True)

    train_data = train_data[['trade_time', 'code', 'nxt1_ret'] + features +
                            regime]
    val_data = val_data[['trade_time', 'code', 'nxt1_ret'] + features + regime]

    train_data = train_data.sort_values('trade_time').reset_index(drop=True)
    val_data = val_data.sort_values('trade_time').reset_index(drop=True)
    # 仅清洗输入；预计算标签的 NaN/Inf 保留给 rl015 判定无效，不补零。
    train_data = _sanitize_frame(train_data, features + regime)
    val_data = _sanitize_frame(val_data, features + regime)
    if train_data['code'].nunique() != 1:
        raise ValueError(
            f"train_data 不是单标的，检测到 {train_data['code'].nunique()} 个 code")
    if val_data['code'].nunique() != 1:
        raise ValueError(
            f"val_data 不是单标的，检测到 {val_data['code'].nunique()} 个 code")
    return train_data, val_data


def train(method, instruments, task_id, period, env_id, trade_id, model_id,
          train_id, feature_id, regime_id):
    pdb.set_trace()
    file_dirs = os.path.join(base_path, method, instruments, 'temp', 'model',
                             str(task_id), str(period), 'rl')

    env_params, trade_params, model_params, train_params, selected_features, min_regime, daily_regime = load_rl_params(
        file_dirs=file_dirs,
        trade_id=trade_id,
        model_id=model_id,
        feature_id=feature_id,
        env_id=env_id,
        train_id=train_id,
        regime_id=regime_id)

    total_params = copy.deepcopy(trade_params)
    total_params.update(env_params)
    total_params.update(model_params)
    total_params.update(train_params)
    total_params.update({'selected_features': selected_features})
    total_params.update({'min_regime': min_regime})
    total_params.update({'daily_regime': daily_regime})

    name = Params.create_tag(total_params)

    output_dir = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl', 'result',
                              str(name))
    tensorboard_dir = os.path.join(tensorboard_path, instruments, str(task_id),
                                   str(name))

    os.makedirs(output_dir, exist_ok=True)
    logger.configure(log_file=os.path.join(output_dir, f"model.log"))
    pdb.set_trace()
    right_basic = PAIRE_TASK[task_id]
    left_train_data, left_val_data = load_data1(
        method=method,
        instruments=instruments,
        period=period,
        task_id=task_id,
        ret_name=trade_params['ret_name'],
        features=selected_features,
        regime=min_regime)
    right_train_data, right_val_data = load_data1(
        method=method,
        instruments=right_basic[0],
        period=period,
        task_id=right_basic[1],
        ret_name=trade_params['ret_name'],
        features=selected_features,
        regime=min_regime)

    train_data = _merge_assets(left_train_data, right_train_data)
    val_data = _merge_assets(left_val_data, right_val_data)

    env_config = {
        'holding_period': int(env_params['holding_period']),
        'reward_scale': float(env_params['reward_scale']),
        'use_tcn': bool(env_params['use_tcn']),
        'default_lookback': int(env_params['default_lookback']),
        'asset_sampling': 'balanced',
        'max_episode_steps': float(env_params['max_episode_steps']),
        'train_scheme': env_params['train_scheme'],
        'softmax_temperature': env_params['softmax_temperature'],
        'seed': env_params['seed']
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
    
    signal_config = {} 
    
    eval_n_episodes = int(train_params['eval_n_episodes'])
    early_stop_patience_evals = int(train_params['early_stop_patience_evals'])
    early_stop_min_evals = int(train_params['early_stop_min_evals'])
    # 短窗口使用平均每步奖励，不能沿用旧累计奖励阈值20.0。
    early_stop_min_delta = float(train_params.get('early_stop_mean_reward_min_delta', 0.0))
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
    pdb.set_trace()
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
            full_eval_freq=int(train_params.get('full_eval_freq', 50000)),
            save_freq=train_params['save_freq'],
            eval_n_episodes=eval_n_episodes,
            val_window_steps=int(train_params.get('val_window_steps', 60)),
            val_windows_per_asset=int(train_params.get('val_windows_per_asset', 160)),
            early_stop_patience_evals=early_stop_patience_evals,
            early_stop_min_evals=early_stop_min_evals,
            early_stop_min_delta=early_stop_min_delta,
            early_stop_start_timesteps=early_stop_start_timesteps,
            verbose=1
        )

    return model, training_info
    
    

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
