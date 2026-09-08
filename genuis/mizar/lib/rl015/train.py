import os, json, gym, pdb, copy
import time
import numpy as np
import pandas as pd
import torch as th
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from kichaos.stable3.sac import SAC
from kichaos.stable3.common.monitor import Monitor
from kichaos.stable3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

from lib.rl015.envs import TradingEnv
from lib.rl015.custom_policy import TCNFeaturesExtractor
from lib.rl015.validation import FixedWindowValidationEnv


class ResetFixWrapper(gym.Wrapper):
    """兼容 reset() 可能返回 (obs, info) 的情况。"""

    def reset(self, **kwargs):
        returned_value = self.env.reset(**kwargs)
        if isinstance(returned_value, tuple) and len(returned_value) == 2:
            observation, _ = returned_value
            return observation
        return returned_value


class EvaluationProgressWrapper(gym.Wrapper):
    """仅报告验证进度；不改变观测、奖励、done 或品种调度。"""

    def __init__(self, env, interval_seconds=15.0):
        super().__init__(env)
        self.interval_seconds = interval_seconds
        self.completed = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.completed = 0
        return obs

    def step(self, action):
        # VecEnv 会在终止后自动 reset；首个 step 才打印，避免虚报下一轮开始。
        if self.completed == 0:
            self.started = self.last_report = time.monotonic()
            self.code = self.env.active_code
            self.total = (self.env.episode_end_offset_exclusive
                          - self.env.episode_start_offset)
            print(f"[VAL_ASSET_START] code={self.code} total={self.total}", flush=True)
        result = self.env.step(action)
        self.completed += 1
        now = time.monotonic()
        done = bool(result[2])
        if done or now - self.last_report >= self.interval_seconds:
            elapsed = now - self.started
            speed = self.completed / max(elapsed, 1e-9)
            eta = max(0, self.total - self.completed) / speed
            label = "VAL_ASSET_END" if done else "VAL_PROGRESS"
            print(f"[{label}] code={self.code} rows={self.completed}/{self.total} "
                  f"({100 * self.completed / max(1, self.total):.1f}%) "
                  f"elapsed={elapsed:.1f}s speed={speed:.1f} rows/s "
                  f"asset_eta={eta:.1f}s", flush=True)
            self.last_report = now
        return result


class QuickValidationCallback(BaseCallback):
    """仅监控：直接评分固定组，不调用EvalCallback，不保存模型、不触发早停。"""

    def __init__(self, env, eval_freq, log_path):
        super().__init__(verbose=0)
        self.env = env
        self.eval_freq = int(eval_freq)
        self.log_path = log_path

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq:
            return True
        group_id = self.env.select_next_group()
        started = time.monotonic()
        rewards = []
        print(f"[EVAL_START] kind=QUICK group={group_id} monitor_only=True "
              f"train_step={self.num_timesteps} episodes={len(self.env.fixed_windows)}", flush=True)
        # predict可能把policy切到eval模式；验证后恢复进入前的训练状态。
        was_training = self.model.policy.training
        try:
            for _ in range(len(self.env.fixed_windows)):
                obs = self.env.reset()
                done, score = False, 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.env.step(action)
                    score += float(reward)
                rewards.append(score)
            mean_reward = float(np.mean(rewards))
            elapsed = time.monotonic() - started
            record = dict(train_step=self.num_timesteps, group_id=group_id,
                          mean_step_reward=mean_reward, windows=len(rewards), elapsed_seconds=elapsed)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.logger.record(f"quick_eval/group_{group_id}/mean_step_reward", mean_reward)
            self.logger.dump(self.num_timesteps)
            print(f"[EVAL_END] kind=QUICK group={group_id} mean_step_reward={mean_reward:.6f} "
                  f"elapsed={elapsed:.1f}s", flush=True)
        except Exception:
            print(f"[EVAL_ERROR] kind=QUICK elapsed={time.monotonic()-started:.1f}s", flush=True)
            raise
        finally:
            self.model.policy.set_training_mode(was_training)
        return True


class MeanStepRewardWrapper(gym.Wrapper):
    """完整品种episode累计后得到该品种平均每步奖励，品种之间等权。"""
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        length = self.env.episode_end_offset_exclusive - self.env.episode_start_offset
        return obs, reward / length, done, info


class _QuickEvaluationLogger:
    """隔离EvalCallback默认的eval/*指标，避免不同组混入完整验证曲线。"""
    def __init__(self, logger, group_id):
        self._logger, self._group_id = logger, group_id

    def record(self, key, value, *args, **kwargs):
        if key.startswith("eval/"):
            key = f"quick_eval/group_{self._group_id}/" + key[5:]
        return self._logger.record(key, value, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._logger, name)


class ProgressEvalCallback(EvalCallback):
    """在原验证回调外报告整轮耗时，保留原评估和最佳模型保存行为。"""

    @property
    def logger(self):
        # 项目的BaseCallback会给self.logger赋值，不是父类只读property。
        # 实际logger由下方setter保存，快速验证读取时才加分组前缀。
        logger = self._callback_logger
        if hasattr(self, "fixed_window_env"):
            return _QuickEvaluationLogger(logger, self.fixed_window_env.active_group + 1)
        return logger

    @logger.setter
    def logger(self, value):
        self._callback_logger = value

    def _on_step(self):
        due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if not due:
            return super()._on_step()
        started = time.monotonic()
        # DummyVecEnv会自动reset，多轮验证前显式归零，确保完全相同的窗口顺序。
        kind = "FULL"
        group_id = None
        if hasattr(self, "fixed_window_env"):
            group_id = self.fixed_window_env.select_next_group()
            kind = f"QUICK group={group_id} monitor_only=True"
        print(f"[EVAL_START] kind={kind} train_step={self.num_timesteps} "
              f"episodes={self.n_eval_episodes}（验证期间训练步数不增长）", flush=True)
        try:
            result = super()._on_step()
        except Exception:
            print(f"[EVAL_ERROR] elapsed={time.monotonic() - started:.1f}s", flush=True)
            raise
        if group_id is not None:
            # 快速验证独立日志带group_id，不与完整验证或其他组混作选择依据。
            record = dict(train_step=self.num_timesteps, group_id=group_id,
                          mean_step_reward=float(self.last_mean_reward))
            with open(self.group_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.logger.record(f"quick_eval/group_{group_id}/mean_step_reward", self.last_mean_reward)
        print(f"[EVAL_END] kind={kind} train_step={self.num_timesteps} "
              f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        return result


class TrainingMetricsCallback(BaseCallback):
    """记录训练中的基础回合统计。"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[float] = []
        self.training_metrics: List[Dict[str, float]] = []
        self._last_progress_time = time.monotonic()
        self._last_progress_step = 0

    def _on_step(self) -> bool:
        now = time.monotonic()
        if now - self._last_progress_time >= 15.0:
            phase = ("经验收集" if self.num_timesteps < self.model.learning_starts
                     else "网络训练")
            print(f"[TRAIN_PROGRESS] step={self.num_timesteps} phase={phase} "
                  f"delta_steps={self.num_timesteps - self._last_progress_step} "
                  f"elapsed={now - self._last_progress_time:.1f}s（含期间验证耗时）",
                  flush=True)
            self._last_progress_time = now
            self._last_progress_step = self.num_timesteps
        if "episode" in self.locals.get("infos", [{}])[0]:
            episode_info = self.locals["infos"][0]["episode"]
            if episode_info is not None:
                self.episode_rewards.append(float(episode_info["r"]))
                self.episode_lengths.append(float(episode_info["l"]))
        return True

    def _on_rollout_end(self) -> bool:
        if self.episode_rewards:
            self.training_metrics.append({
                "step":
                float(self.num_timesteps),
                "mean_episode_reward":
                float(np.mean(self.episode_rewards[-10:])),
                "mean_episode_length":
                float(np.mean(self.episode_lengths[-10:])),
            })
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
        start_timesteps: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.max_no_improvement_evals = int(max_no_improvement_evals)
        self.min_evals = int(min_evals)
        self.min_delta = float(min_delta)
        self.start_timesteps = int(start_timesteps)
        self._last_eval_count = 0
        self._best_seen = -np.inf
        self._no_improve_count = 0
        self._monitoring_started = False
        self._monitor_eval_count = 0

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
        delta_eval = int(eval_count - self._last_eval_count)
        best_reward = float(
            getattr(self.eval_callback, "best_mean_reward", -np.inf))
        if not np.isfinite(best_reward):
            best_reward = -np.inf
        if not self._monitoring_started:
            if self.num_timesteps < self.start_timesteps:
                self._last_eval_count = eval_count
                return True
            self._monitoring_started = True
            self._best_seen = best_reward
            self._no_improve_count = 0
            self._monitor_eval_count = 1
            self._last_eval_count = eval_count
            if self.verbose > 0:
                print(
                    "[EARLY STOP] 开始监控: "
                    f"start_timesteps={self.start_timesteps}, "
                    f"timestep={self.num_timesteps}, best_mean_reward={best_reward:.6f}"
                )
            return True
        self._monitor_eval_count += delta_eval
        if best_reward > (self._best_seen + self.min_delta):
            self._best_seen = best_reward
            self._no_improve_count = 0
        else:
            self._no_improve_count += delta_eval
        self._last_eval_count = eval_count
        if (self._monitor_eval_count >= self.min_evals
                and self._no_improve_count >= self.max_no_improvement_evals):
            if self.verbose > 0:
                print("[EARLY STOP] 连续"
                      f" {self._no_improve_count} 次评估无提升，"
                      f"在 timestep={self.num_timesteps} 提前停止训练。")
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
    # 只给输入特征补零；未来收益的缺失必须由环境识别为无效标签。
    numeric_cols = list(features)
    existed_cols = [c for c in numeric_cols if c in out.columns]
    if not existed_cols:
        return out
    out[existed_cols] = out[existed_cols].apply(pd.to_numeric, errors="coerce")
    bad_mask = ~np.isfinite(out[existed_cols].to_numpy(dtype=np.float64))
    bad_count = int(bad_mask.sum())
    if bad_count > 0:
        print(f"[WARN] 检测到 {bad_count} 个非有限值(NaN/Inf)，已用 0.0 替换。")
    out[existed_cols] = out[existed_cols].replace([np.inf, -np.inf],
                                                  np.nan).fillna(0.0)
    return out


def create_env(df: pd.DataFrame,
               mode: str,
               features: List[str],
               env_config: Dict[str, Any],
               signal_config: Optional[Any] = None):
    if "nxt1_ret" not in df.columns:
        raise ValueError("训练/验证数据必须包含 'nxt1_ret' 列")
    df = _sanitize_dataframe(df, features)
    env_config = dict(env_config)
    env_config['mode'] = mode
    config = {"env_config": env_config, "signal_config": signal_config}

    return TradingEnv(df=df, features=features, config=config)


def train_model(train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                features: List[str],
                env_config: Dict[str, Any],
                sac_config: Dict[str, Any],
                signal_config: Optional[Any] = None,
                output_dir: str = "",
                tensorboard_dir: str = "",
                model_dir: str = "",
                total_timesteps: int = 100000,
                eval_freq: int = 10000,
                eval_n_episodes: int = 1,
                save_freq: int = 50000,
                early_stop_patience_evals: int = 6,
                early_stop_min_evals: int = 6,
                early_stop_min_delta: float = 0.0,
                early_stop_start_timesteps: int = 0,
                enable_early_stop: bool = True,
                verbose: int = 1,
                val_window_steps: int = 60,
                val_windows_per_asset: int = 160,
                full_eval_freq: int = 50000) -> Tuple[SAC, Dict[str, Any]]:
    if eval_freq <= 0 or full_eval_freq <= 0:
        raise ValueError("eval_freq 和 full_eval_freq 必须大于0")
    if not output_dir:
        raise ValueError("output_dir 不能为空")
    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    tensorboard_dir = os.path.join(
        output_dir,
        "tensorboard") if len(tensorboard_dir) == 0 else tensorboard_dir
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    train_env = create_env(
        mode='train',
        df=train_df,
        features=features,
        env_config=env_config,
        signal_config=signal_config,
    )
    
    # 【多资产修改 11：尺度传递】先创建训练环境，取出各品种收益标准差，
    # 再传入验证环境。下方 config.json 保存此 env_config，预测时原样复用。
    # 若调用方已提供 ret_scale_by_code，训练环境会优先使用提供的尺度。
    # 标准化尺度只能从训练集估计；验证和测试沿用该尺度。
    
    env_config = dict(env_config)
    env_config["ret_scale_by_code"] = dict(train_env.ret_scale_by_code)
    print(f"按品种训练集收益标准差: {env_config['ret_scale_by_code']}")

    # 【训练流程补齐】验证复用训练尺度，不在验证集重新估计。
    fixed_env = FixedWindowValidationEnv(
        df=_sanitize_dataframe(val_df, features), features=features,
        config={"env_config": dict(env_config, mode="val"), "signal_config": signal_config},
        window_steps=val_window_steps, windows_per_asset=val_windows_per_asset,
    )
    # 窗口方案下episode数由实际固定窗口数决定，不再由品种数决定。
    eval_n_episodes = len(fixed_env.fixed_windows)
    with open(os.path.join(log_dir, "validation_windows.json"), "w", encoding="utf-8") as f:
        json.dump(fixed_env.window_records(), f, ensure_ascii=False, indent=2)
    print(f"[VAL_FIXED] groups=4 windows_per_group={eval_n_episodes} steps_per_window={val_window_steps} "
          f"total_rows={eval_n_episodes * val_window_steps} metric=mean_step_reward", flush=True)
    train_env = Monitor(ResetFixWrapper(train_env),
                        filename=os.path.join(log_dir, "train_monitor.csv"))
    full_selection_env = create_env(df=val_df, mode="val", features=features,
                                    env_config=env_config, signal_config=signal_config)
    full_eval_episodes = full_selection_env.n_assets
    val_env = Monitor(MeanStepRewardWrapper(EvaluationProgressWrapper(
        ResetFixWrapper(full_selection_env))), filename=os.path.join(log_dir, "val_monitor.csv"))

    sac_config_clean = copy.deepcopy(_sanitize_sac_config(sac_config))
    # 单步预测标签已经包含未来五分钟收益，不需要再折扣累加后续标签。
    sac_config_clean.setdefault("gamma", 0.0)
    if env_config.get("use_tcn", False):
        policy_kwargs = dict(sac_config_clean.get("policy_kwargs", {}))
        policy_kwargs["features_extractor_class"] = TCNFeaturesExtractor
        # 网络参数可通过 policy_kwargs.features_extractor_kwargs 显式指定。
        policy_kwargs.setdefault("features_extractor_kwargs", {})
        sac_config_clean["policy_kwargs"] = policy_kwargs

    model = SAC(
        policy="MlpPolicy", env=train_env, tensorboard_log=tensorboard_dir,
        verbose=verbose, seed=env_config.get("seed"), **sac_config_clean,
    )
    # set_random_seed() 为复现默认关闭了 CuDNN autotune；本任务输入形状固定，
    # CUDA 下允许 CuDNN 选择更快卷积实现。代价是同种子不保证逐位一致。
    if model.device.type == "cuda":
        th.backends.cudnn.deterministic = False
        th.backends.cudnn.benchmark = True
    print(
        "[SAC_RUNTIME] "
        f"device={model.device} batch_size={model.batch_size} "
        f"train_freq={model.train_freq} gradient_steps={model.gradient_steps} "
        f"learning_starts={model.learning_starts} gamma={model.gamma} "
        f"cudnn_deterministic={th.backends.cudnn.deterministic} "
        f"cudnn_benchmark={th.backends.cudnn.benchmark}",
        flush=True,
    )
    quick_callback = QuickValidationCallback(
        fixed_env, eval_freq=eval_freq,
        log_path=os.path.join(log_dir, "quick_validation.jsonl"),
    )
    eval_callback = ProgressEvalCallback(
        val_env, best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"), n_eval_episodes=full_eval_episodes,
        eval_freq=full_eval_freq, deterministic=True, render=False, verbose=verbose,
    )
    callbacks = [quick_callback, eval_callback]
    if enable_early_stop:
        callbacks.append(EarlyStopOnNoImprovement(
            eval_callback=eval_callback,
            max_no_improvement_evals=early_stop_patience_evals,
            min_evals=early_stop_min_evals, min_delta=early_stop_min_delta,
            start_timesteps=early_stop_start_timesteps, verbose=verbose,
        ))
    callbacks.append(CheckpointCallback(
        save_freq=save_freq, save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="sac_model", verbose=verbose,
    ))
    metrics_callback = TrainingMetricsCallback(verbose=verbose)
    callbacks.append(metrics_callback)

    # JSON 只记录网络类的名称；真正的类及网络权重由 SAC.save 保存，
    # SignalGenerator 使用 SAC.load 恢复，不把 JSON 中的字符串当作类调用。
    config_info = {
        "env_config": env_config, "sac_config": sac_config_clean,
        "signal_config": signal_config, "features": features,
        "total_timesteps": total_timesteps, "eval_n_episodes": eval_n_episodes,
        "validation": {"window_steps": val_window_steps,
                       "groups": 4, "quick_eval_freq": eval_freq,
                       "full_eval_freq": full_eval_freq,
                       "selection_source": "full_validation_only",
                       "requested_windows_per_asset": val_windows_per_asset,
                       "metric": "mean_step_reward"},
        "early_stop": {
            "enabled": bool(enable_early_stop),
            "patience_evals": int(early_stop_patience_evals),
            "min_evals": int(early_stop_min_evals),
            "min_delta": float(early_stop_min_delta),
            "start_timesteps": int(early_stop_start_timesteps),
        },
        "train_size": len(train_df), "val_size": len(val_df),
        "training_date": datetime.now().isoformat(),
    }

    def json_default(value):
        if isinstance(value, type):
            return f"{value.__module__}.{value.__qualname__}"
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"配置中存在无法 JSON 序列化的类型: {type(value).__name__}")

    # 训练前保存，最佳模型/中途检查点也能找到对应的收益尺度和输入配置。
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False, default=json_default)
    print(f"开始训练，总步数: {total_timesteps}，快速验证间隔: {eval_freq}，完整验证间隔: {full_eval_freq}")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, log_interval=4)
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    # 仅训练结束时完整复核；不使用测试集，也不据此再次训练或替换最佳模型。
    best_path = os.path.join(model_dir, "best_model", "best_model")
    if os.path.isfile(best_path + ".zip") or os.path.isfile(best_path):
        review_model = SAC.load(best_path)
        reviewed_kind = "best"
    else:
        print("[FULL_VAL] 未生成最佳模型（可能尚未触发评估），明确复核final模型", flush=True)
        review_model, reviewed_kind = model, "final"
    full_env = EvaluationProgressWrapper(ResetFixWrapper(create_env(
        df=val_df, mode="val", features=features, env_config=env_config,
        signal_config=signal_config)))
    full_results = []
    print(f"[FULL_VAL_START] model={reviewed_kind} 训练结束完整复核", flush=True)
    for _ in range(full_env.env.n_assets):
        obs = full_env.reset()
        code = full_env.env.active_code
        reward_sum, count, done = 0.0, 0, False
        while not done:
            action, _ = review_model.predict(obs, deterministic=True)
            obs, reward, done, _ = full_env.step(action)
            reward_sum += float(reward)
            count += 1
        full_results.append(dict(code=code, rows=count, mean_step_reward=reward_sum / count))
    full_path = os.path.join(log_dir, "full_validation.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(dict(model=reviewed_kind, assets=full_results,
                       mean_step_reward=float(np.mean([r["mean_step_reward"] for r in full_results]))),
                  f, ensure_ascii=False, indent=2)
    full_env.close()
    print(f"[FULL_VAL_END] {full_path}", flush=True)
    if metrics_callback.training_metrics:
        with open(os.path.join(log_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_callback.training_metrics, f, indent=2, ensure_ascii=False)
    print(f"最终模型已保存到: {final_model_path}")
    training_info = {
        "model_path": final_model_path,
        "best_model_path": os.path.join(model_dir, "best_model", "best_model"),
        "config_path": config_path, "log_dir": log_dir,
        "tensorboard_dir": tensorboard_dir,
        "training_metrics": metrics_callback.training_metrics,
        "full_validation_path": full_path,
    }
    return model, training_info
