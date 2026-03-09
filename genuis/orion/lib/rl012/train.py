import os, json, copy
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

from lib.rl012.envs import TradingEnv
from lib.rl012.signal import Config
from lib.rl012.custom_policy import CrossSectionalSACPolicy

from kichaos.stable3.sac import SAC
from kichaos.stable3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from kichaos.stable3.common.monitor import Monitor
from kdutils.logger import logger


class TrainingMetricsCallback(BaseCallback):
    """训练指标回调"""
    
    def __init__(self, verbose=0, log_dir=None):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        
    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            episode_info = info.get('episode')
            if episode_info is not None:
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
        return True
    
    def _on_rollout_end(self) -> bool:
        if len(self.episode_rewards) > 0:
            metrics = {
                'step': self.num_timesteps,
                'mean_episode_reward': float(np.mean(self.episode_rewards[-10:])),
                'mean_episode_length': float(np.mean(self.episode_lengths[-10:])),
            }
            self.training_metrics.append(metrics)
        return True


def create_env(
    df: pd.DataFrame,
    features: List[str],
    config: Dict[str, Any],
    signal_config: Config,
) -> TradingEnv:
    """创建交易环境"""
    env = TradingEnv(
        df=df,
        features=features,
        subset_size=config['subset_size'],
        episode_len=config['episode_len'],
        seed=config['seed'],
        reward_scale=config['reward_scale'],
        signal_config=signal_config,
        ic_scale=config['ic_scale'],
        negative_ic_penalty=config['negative_ic_penalty'],
        use_turnover_proxy=config['use_turnover_proxy'],
        turnover_proxy_coef=config['turnover_proxy_coef'],
        use_fee_in_reward=config['use_fee_in_reward'],
        fee_coef=config['fee_coef']
    )
    return env


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    env_config: Dict[str, Any],
    sac_config: Dict[str, Any],
    signal_config: Config,
    output_dir: str,
    total_timesteps: int,
    eval_freq: int,
    eval_n_episodes: int = 5,
    save_freq: int = 50000,
    verbose: int = 1,
    use_custom_policy: bool = False,
) -> Tuple[SAC, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    log_dir = os.path.join(output_dir, 'logs')
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 创建训练环境
    raw_train_env = create_env(train_df, features, env_config, signal_config)
    train_env = Monitor(raw_train_env, filename=os.path.join(log_dir, 'train_monitor'))
    
    # 创建校验环境
    raw_val_env = create_env(val_df, features, env_config, signal_config)
    val_env = Monitor(raw_val_env, filename=os.path.join(log_dir, 'val_monitor'))
    
    logger.info(f"训练环境: {raw_train_env}")
    logger.info(f"校验环境: {raw_val_env}")
    
    # ── 选择 Policy ──
    working_sac_config = copy.deepcopy(sac_config)
    
    if use_custom_policy:
        policy_class = CrossSectionalSACPolicy
        
        if 'policy_kwargs' not in working_sac_config:
            working_sac_config['policy_kwargs'] = {}
        working_sac_config['policy_kwargs']['n_assets'] = raw_train_env.subset_size
        working_sac_config['policy_kwargs']['n_stock_features'] = len(features)
        
        logger.info(f"使用 CrossSectionalSACPolicy (n_assets={raw_train_env.subset_size}, n_features={len(features)})")
    else:
        policy_class = 'MlpPolicy'
        logger.info(f"使用 MlpPolicy")

    model = SAC(
        policy=policy_class,
        env=train_env,
        tensorboard_log=tensorboard_dir,
        verbose=verbose,
        seed=env_config['seed'],
        **working_sac_config
    )
    
    # ── 回调 ──
    callbacks = []
    
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=os.path.join(model_dir, 'best_model'),
        log_path=os.path.join(log_dir, 'eval'),
        eval_freq=eval_freq,
        n_eval_episodes=eval_n_episodes,
        deterministic=True,
        render=False,
        verbose=verbose
    )
    callbacks.append(eval_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(model_dir, 'checkpoints'),
        name_prefix='sac_r012',
    )
    callbacks.append(checkpoint_callback)

    metrics_callback = TrainingMetricsCallback(verbose=verbose, log_dir=log_dir)
    callbacks.append(metrics_callback)
    
    # ── 保存配置 ──
    safe_sac_config = copy.deepcopy(working_sac_config)
    if 'policy_kwargs' in safe_sac_config:
        pk = safe_sac_config['policy_kwargs']
        if 'features_extractor_class' in pk and hasattr(pk['features_extractor_class'], '__name__'):
            pk['features_extractor_class'] = pk['features_extractor_class'].__name__
    safe_sac_config = {
        k: str(v) if not isinstance(v, (int, float, bool, str, type(None), dict, list)) else v 
        for k, v in safe_sac_config.items()
    }
    
    config_to_save = {
        'env_config': env_config,
        'sac_config': safe_sac_config,
        'signal_config': {
            'min_weight': signal_config.min_weight,
            'max_weight': signal_config.max_weight,
            'normalize': signal_config.normalize,
            'top_k': signal_config.top_k,
            'cost_rate': signal_config.cost_rate,
            'stamp_duty': signal_config.stamp_duty,
            'turnover_penalty': signal_config.turnover_penalty,
            'rebalance_window': signal_config.rebalance_window,
            'softmax_temperature': signal_config.softmax_temperature,
        },
        'features': features,
        'use_custom_policy': use_custom_policy,
        'total_timesteps': total_timesteps,
        'eval_n_episodes': eval_n_episodes,
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'training_start': datetime.now().isoformat(),
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2, default=str)
        
    # ── 训练 ──
    logger.info(f"开始训练 截面选股模型 (r012)...")
    logger.info(f"  数据行数(训练): {len(train_df)}")
    logger.info(f"  数据行数(校验): {len(val_df)}")
    logger.info(f"  subset_size: {raw_train_env.subset_size}")
    logger.info(f"  episode_len: {raw_train_env.episode_len}")
    logger.info(f"  动作空间: {raw_train_env.action_space}")
    logger.info(f"  观测空间: {raw_train_env.observation_space}")
    logger.info(f"  总步数: {total_timesteps}")
    logger.info(f"  policy: {'CrossSectionalSACPolicy' if use_custom_policy else 'MlpPolicy'}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10
    )
    
    # ── 保存模型 ──
    final_model_path = os.path.join(model_dir, 'final_model')
    model.save(final_model_path)
    
    best_model_path = os.path.join(model_dir, 'best_model', 'best_model.zip')
    training_info = {
        'model_path': final_model_path,
        'best_model_path': best_model_path,
        'config_path': config_path,
        'output_dir': output_dir,
        'total_timesteps': total_timesteps,
        'use_custom_policy': use_custom_policy,
    }
    
    logger.info(f"训练完成！最终模型: {final_model_path}")
    
    return model, training_info
