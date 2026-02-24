import os, gym, json, pdb
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from kichaos.stable3.sac import SAC
from kichaos.stable3.common.monitor import Monitor
from kichaos.stable3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

from lib.rl001.signal import Config
from lib.rl001.trade_env import TradingEnv


class ResetFixWrapper(gym.Wrapper):
    """包装器：处理 reset() 返回 (observation, info) 的情况"""
    
    def reset(self, **kwargs):
        returned_value = self.env.reset(**kwargs)
        if isinstance(returned_value, tuple) and len(returned_value) == 2:
            # 返回 (observation, info)，但 DummyVecEnv 只需要 observation
            observation, info = returned_value
            return observation
        else:
            # 已经只返回 observation
            return returned_value
       
class TrainingMetricsCallback(BaseCallback):
    """训练指标回调"""
    
    def __init__(self, verbose=0, log_dir=None):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        
    def _on_step(self) -> bool:
        # 记录回合信息
        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_info = self.locals['infos'][0]['episode']
            if episode_info is not None:
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
        return True
    
    def _on_rollout_end(self) -> bool:
        # 记录训练指标
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
               signal_config: Config):
    env = TradingEnv(
        df=df,
        features=features,
        mode=config.get('mode', 'UNLOCK'),
        holding_period=config.get('holding_period', 15),
        max_pairs=config.get('max_pairs', 50),
        max_allowed_position=config.get('max_allowed_position', 10),
        use_cooldown=config.get('use_cooldown', True),
        cooldown_steps=config.get('cooldown_steps', 3),
        include_market_features=config.get('include_market_features', True),
        volatility_window=config.get('volatility_window', 60),
        volume_window=config.get('volume_window', 60),
        masking_threshold_multiplier=config.get('masking_threshold_multiplier', 1.0),
        episode_len=config.get('episode_len', 500),
        seed=config.get('seed', None),
        cost_rate=config.get('cost_rate', 0.0001),
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
    
    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 创建训练环境
    train_env = create_env(train_df, features, env_config, signal_config)
    train_env = ResetFixWrapper(train_env)  # 修复 reset() 返回值
    train_env = Monitor(train_env, filename=os.path.join(log_dir, 'train_monitor.csv'))

    # 创建校验环境
    val_env = create_env(val_df, features, env_config, signal_config)
    val_env = ResetFixWrapper(val_env)  # 修复 reset() 返回值
    val_env = Monitor(val_env, filename=os.path.join(log_dir, 'val_monitor.csv'))


    model = SAC(
        policy='MlpPolicy',
        env=train_env,
        tensorboard_log=tensorboard_dir,
        verbose=verbose,
        seed=env_config.get('seed', None),
        **sac_config
    )
    # 创建回调函数
    callbacks = []
    
    # 评估回调
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
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(model_dir, 'checkpoints'),
        name_prefix='sac_model',
        verbose=verbose
    )
    callbacks.append(checkpoint_callback)

    # 训练指标回调
    metrics_callback = TrainingMetricsCallback(
        verbose=verbose,
        log_dir=log_dir
    )
    callbacks.append(metrics_callback)
    
    # 开始训练
    print(f"开始训练，总步数: {total_timesteps}")
    print(f"训练集大小: {len(train_df)}, 校验集大小: {len(val_df)}")
    print(f"特征数量: {len(features)}")
    print(f"输出目录: {output_dir}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=4
    )
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'final_model')
    model.save(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 保存配置信息
    config_info = {
        'env_config': env_config,
        'sac_config': sac_config,
        'signal_config': {
            'threshold_mode': signal_config.threshold_mode,
            'threshold': signal_config.threshold,
            'base_threshold': signal_config.base_threshold,
            'threshold_k': signal_config.threshold_k,
            'threshold_min': signal_config.threshold_min,
            'threshold_max': signal_config.threshold_max,
            'base_cost': signal_config.base_cost,
            'cost_multiplier': signal_config.cost_multiplier,
            'cost_mode': signal_config.cost_mode,
        },
        'features': features,
        'total_timesteps': total_timesteps,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'training_date': datetime.now().isoformat()
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"配置信息已保存到: {config_path}")
    
    # 保存训练指标
    if metrics_callback.training_metrics:
        metrics_path = os.path.join(log_dir, 'training_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_callback.training_metrics, f, indent=2)
        print(f"训练指标已保存到: {metrics_path}")
    
    training_info = {
        'model_path': final_model_path,
        'best_model_path': os.path.join(model_dir, 'best_model', 'best_model'),
        'config_path': config_path,
        'log_dir': log_dir,
        'tensorboard_dir': tensorboard_dir,
        'training_metrics': metrics_callback.training_metrics
    }
    
    return model, training_info