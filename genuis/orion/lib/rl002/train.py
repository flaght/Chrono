import os, json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

from lib.rl002.envs import TradingEnv
from lib.rl002.signal import Config

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
                'mean_episode_reward': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards),
                'mean_episode_length': np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else np.mean(self.episode_lengths),
            }
            self.training_metrics.append(metrics)
        return True
    
def create_env(df: pd.DataFrame,
               features: List[str],
               config: Dict[str, Any],
               signal_config: Config) -> TradingEnv:
    """创建交易环境"""
    env = TradingEnv(
        df=df,
        features=features,
        n_assets=config.get('n_assets', 0),
        episode_len=config.get('episode_len', 500),
        seed=config.get('seed', None),
        reward_scale=config.get('reward_scale', 10000.0),
        signal_config=signal_config
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
    total_timesteps: int = 100000,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    verbose: int = 1
) -> Tuple[SAC, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    log_dir = os.path.join(output_dir, 'logs')
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    
    # 创建训练环境
    train_env = create_env(train_df, features, env_config, signal_config)
    train_env = Monitor(train_env, filename=os.path.join(log_dir, 'train_monitor'))
    
    logger.info(f"训练环境: {train_env}")
    
    
    # 创建校验环境
    val_env = create_env(val_df, features, env_config, signal_config)
    val_env = Monitor(val_env, filename=os.path.join(log_dir, 'val_monitor'))
    
    # 创建 SAC 模型
    model = SAC(
        policy='MlpPolicy',
        env=train_env,
        tensorboard_log=tensorboard_dir,
        verbose=verbose,
        seed=env_config.get('seed', None),
        **sac_config
    )
    # 回调
    callbacks = []
    
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=os.path.join(model_dir, 'best_model'),
        log_path=os.path.join(log_dir, 'eval'),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=verbose
    )
    callbacks.append(eval_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(model_dir, 'checkpoints'),
        name_prefix='sac_stock_model',
    )
    
    callbacks.append(checkpoint_callback)

    metrics_callback = TrainingMetricsCallback(verbose=verbose, log_dir=log_dir)
    callbacks.append(metrics_callback)
    
    # 保存配置
    config_to_save = {
        'env_config': env_config,
        'sac_config': {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v 
                       for k, v in sac_config.items()},
        'signal_config': {
            'min_weight': signal_config.min_weight,
            'max_weight': signal_config.max_weight,
            'normalize': signal_config.normalize,
            'top_k': signal_config.top_k,
            'cost_rate': signal_config.cost_rate,
            'stamp_duty': signal_config.stamp_duty,
            'turnover_penalty': signal_config.turnover_penalty,
            'rebalance_window': signal_config.rebalance_window,
        },
        'features': features,
        'total_timesteps': total_timesteps,
        'training_start': datetime.now().isoformat(),
    }
    
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2, default=str)
        
    # 开始训练
    logger.info(f"开始训练 A股截面选股模型...")
    logger.info(f"  股票数量: {train_env.n_assets}")
    logger.info(f"  动作空间: {train_env.action_space}")
    logger.info(f"  总步数: {total_timesteps}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10
    )
    
    # 保存
    final_model_path = os.path.join(model_dir, 'final_model')
    model.save(final_model_path)
    
    best_model_path = os.path.join(model_dir, 'best_model', 'best_model.zip')
    training_info = {
        'model_path': final_model_path,
        'best_model_path': best_model_path,
        'config_path': config_path,
        'output_dir': output_dir,
        'total_timesteps': total_timesteps,
    }
    
    logger.info(f"训练完成！最终模型: {final_model_path}")
    
    return model, training_info
    
    