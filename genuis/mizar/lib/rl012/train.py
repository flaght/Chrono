import os, json, gym
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from kichaos.stable3.sac import SAC
from kichaos.stable3.common.monitor import Monitor
from kichaos.stable3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

from lib.rl012.envs import TradingEnv

class ResetFixWrapper(gym.Wrapper):
    """兼容 reset() 可能返回 (obs, info) 的情况。"""
    def reset(self, **kwargs):
        returned_value = self.env.reset(**kwargs)
        if isinstance(returned_value, tuple) and len(returned_value) == 2:
            observation, _ = returned_value
            return observation
        return returned_value
    
class TrainingMetricsCallback(BaseCallback):
    """记录训练中的基础回合统计。"""
       
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[float] = []
        self.training_metrics: List[Dict[str, float]] = []
        
    def _on_step(self) -> bool:
        if "episode" in self.locals.get("infos", [{}])[0]:
            episode_info = self.locals["infos"][0]["episode"]
            if episode_info is not None:
                self.episode_rewards.append(float(episode_info["r"]))
                self.episode_lengths.append(float(episode_info["l"]))
        return True
    def _on_rollout_end(self) -> bool:
        if self.episode_rewards:
            self.training_metrics.append(
                {
                    "step": float(self.num_timesteps),
                    "mean_episode_reward": float(np.mean(self.episode_rewards[-10:])),
                    "mean_episode_length": float(np.mean(self.episode_lengths[-10:])),
                }
            )
        return True
class EarlyStopOnNoImprovement(BaseCallback):
    """
    在评估回调长期无提升时提前停止训练，避免无效训练。
    """
    def __init__(
        self,
        eval_callback: EvalCallback,
        max_no_improvement_evals: int = 6,
        min_evals: int = 6,
        min_delta: float = 0.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.max_no_improvement_evals = int(max_no_improvement_evals)
        self.min_evals = int(min_evals)
        self.min_delta = float(min_delta)
        self._last_eval_count = 0
        self._best_seen = -np.inf
        self._no_improve_count = 0
    def _get_eval_count(self) -> int:
        for attr in ("evaluations_timesteps", "evaluations_results"):
            val = getattr(self.eval_callback, attr, None)
            if isinstance(val, list):
                return int(len(val))
        eval_freq = int(getattr(self.eval_callback, "eval_freq", 0) or 0)
        n_calls = int(getattr(self.eval_callback, "n_calls", 0) or 0)
        if eval_freq > 0:
            return int(n_calls // eval_freq)
        return 0
    def _on_step(self) -> bool:
        eval_count = self._get_eval_count()
        if eval_count <= self._last_eval_count:
            return True
        best_reward = float(getattr(self.eval_callback, "best_mean_reward", -np.inf))
        if not np.isfinite(best_reward):
            best_reward = -np.inf
        if best_reward > (self._best_seen + self.min_delta):
            self._best_seen = best_reward
            self._no_improve_count = 0
        else:
            self._no_improve_count += int(eval_count - self._last_eval_count)
        self._last_eval_count = eval_count
        if (
            eval_count >= self.min_evals
            and self._no_improve_count >= self.max_no_improvement_evals
        ):
            if self.verbose > 0:
                print(
                    "[EARLY STOP] 连续"
                    f" {self._no_improve_count} 次评估无提升，"
                    f"在 timestep={self.num_timesteps} 提前停止训练。"
                )
            return False
        return True
    

def _sanitize_sac_config(sac_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    过滤掉 SAC 不支持的参数，避免例如 log_dir 透传导致初始化报错。
    """
    allowed_keys = {
        "learning_rate",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "tau",
        "gamma",
        "train_freq",
        "gradient_steps",
        "ent_coef",
        "target_update_interval",
        "policy_kwargs",
    }
    return {k: v for k, v in dict(sac_config).items() if k in allowed_keys}


def _sanitize_dataframe(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    清洗训练数据中的 NaN/Inf，防止观测进入网络后产生 NaN 梯度。
    """
    out = df.copy()
    numeric_cols = list(features) + ["nxt1_ret"]
    existed_cols = [c for c in numeric_cols if c in out.columns]
    if not existed_cols:
        return out
    out[existed_cols] = out[existed_cols].apply(pd.to_numeric, errors="coerce")
    bad_mask = ~np.isfinite(out[existed_cols].to_numpy(dtype=np.float64))
    bad_count = int(bad_mask.sum())
    if bad_count > 0:
        print(f"[WARN] 检测到 {bad_count} 个非有限值(NaN/Inf)，已用 0.0 替换。")
    out[existed_cols] = out[existed_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def create_env(
    df: pd.DataFrame,
    features: List[str],
    env_config:Dict[str, Any],
    signal_config: Optional[Any] = None
):
    if "nxt1_ret" not in df.columns:
        raise ValueError("训练/验证数据必须包含 'nxt1_ret' 列")
    df = _sanitize_dataframe(df, features)
    config = {
        "env_config": env_config,
        "signal_config": signal_config
    }
    
    return TradingEnv(df=df, features=features, config=config)

def train_model(train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    env_config: Dict[str, Any],
    sac_config: Dict[str, Any],
    signal_config: Optional[Any] = None,
    output_dir: str = "",
    total_timesteps: int = 100000,
    eval_freq: int = 10000,
    eval_n_episodes: int = 1,
    save_freq: int = 50000,
    early_stop_patience_evals: int = 6,
    early_stop_min_evals: int = 6,
    early_stop_min_delta: float = 0.0,
    enable_early_stop: bool = True,
    verbose: int = 1)-> Tuple[SAC, Dict[str, Any]]:
    if not output_dir:
        raise ValueError("output_dir 不能为空")
    
    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    train_env = create_env(
        df=train_df,
        features=features,
        env_config=env_config,
        signal_config=signal_config,
    )
    train_env = ResetFixWrapper(train_env)
    train_env = Monitor(train_env, filename=os.path.join(log_dir, "train_monitor.csv"))
    
    
    
    val_env = create_env(
        df=val_df,
        features=features,
        env_config=env_config,
        signal_config=signal_config,
    )
    val_env = ResetFixWrapper(val_env)
    val_env = Monitor(val_env, filename=os.path.join(log_dir, "val_monitor.csv"))
    
    sac_config_clean = _sanitize_sac_config(sac_config)
    
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        tensorboard_log=tensorboard_dir,
        verbose=verbose,
        seed=env_config.get("seed"),
        **sac_config_clean,
    )
    
    # 创建回调函数
    callbacks = []
    
    # 评估回调
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=os.path.join(model_dir, 'best_model'),
        log_path=os.path.join(log_dir, 'eval'),
        n_eval_episodes=int(eval_n_episodes),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=verbose
    )
    callbacks.append(eval_callback)
    if enable_early_stop:
        early_stop_callback = EarlyStopOnNoImprovement(
            eval_callback=eval_callback,
            max_no_improvement_evals=early_stop_patience_evals,
            min_evals=early_stop_min_evals,
            min_delta=early_stop_min_delta,
            verbose=verbose,
        )
        callbacks.append(early_stop_callback)
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(model_dir, 'checkpoints'),
        name_prefix='sac_model',
        verbose=verbose
    )
    callbacks.append(checkpoint_callback)
    
    # 训练指标回调
    metrics_callback = TrainingMetricsCallback(verbose=verbose)
    callbacks.append(metrics_callback)
    
    # 开始训练
    print(f"开始训练，总步数: {total_timesteps}")
    print(f"训练集大小: {len(train_df)}, 校验集大小: {len(val_df)}")
    print(f"特征数量: {len(features)}")
    print(f"输出目录: {output_dir}")
    if enable_early_stop:
        print(
            "早停已启用: "
            f"patience_evals={early_stop_patience_evals}, "
            f"min_evals={early_stop_min_evals}, min_delta={early_stop_min_delta}"
        )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=4
    )
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'final_model')
    model.save(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    config_info = {
        "env_config": env_config,
        "sac_config": sac_config_clean,
        "signal_config": signal_config,
        "features": features,
        "total_timesteps": total_timesteps,
        "eval_n_episodes": int(eval_n_episodes),
        "early_stop": {
            "enabled": bool(enable_early_stop),
            "patience_evals": int(early_stop_patience_evals),
            "min_evals": int(early_stop_min_evals),
            "min_delta": float(early_stop_min_delta),
        },
        "train_size": len(train_df),
        "val_size": len(val_df),
        "training_date": datetime.now().isoformat(),
    }
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"配置信息已保存到: {config_path}")
    if metrics_callback.training_metrics:
        metrics_path = os.path.join(log_dir, "training_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_callback.training_metrics, f, indent=2, ensure_ascii=False)
        print(f"训练指标已保存到: {metrics_path}")
    training_info = {
        "model_path": final_model_path,
        "best_model_path": os.path.join(model_dir, "best_model", "best_model"),
        "config_path": config_path,
        "log_dir": log_dir,
        "tensorboard_dir": tensorboard_dir,
        "training_metrics": metrics_callback.training_metrics,
    }
    return model, training_info
    
    
    
    
    