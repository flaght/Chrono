"""rl016 训练：负 MSE 奖励，快速 IC 监控，完整 IC 选模。"""

import copy
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import pandas as pd
import torch as th

from kichaos.stable3.common.callbacks import BaseCallback, CheckpointCallback
from kichaos.stable3.common.monitor import Monitor
from kichaos.stable3.sac import SAC

from lib.rl016.custom_policy import TCNFeaturesExtractor
from lib.rl016.envs import TradingEnv
from lib.rl016.inference import predict_environment
from lib.rl016.metrics import evaluate_predictions
from lib.rl016.validation import FixedWindowValidationEnv


class ResetFixWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        value = self.env.reset(**kwargs)
        return value[0] if isinstance(value, tuple) and len(value) == 2 else value


def _sanitize_dataframe(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    out = df.copy()
    missing = {"trade_time", "code", "nxt1_ret", *features} - set(out.columns)
    if missing:
        raise ValueError(f"数据缺少字段: {sorted(missing)}")
    out[features] = out[features].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(out[features].to_numpy(dtype=np.float64))
    if int(bad.sum()):
        print(f"[WARN] 特征中发现 {int(bad.sum())} 个 NaN/Inf，已填充为 0.0")
    out[features] = out[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def create_env(df: pd.DataFrame, mode: str, features: List[str],
               env_config: Dict[str, Any], signal_config=None) -> TradingEnv:
    cfg = dict(env_config, mode=mode)
    return TradingEnv(_sanitize_dataframe(df, features), features,
                      {"env_config": cfg, "signal_config": signal_config or {}})


def _sanitize_sac_config(config: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"learning_rate", "buffer_size", "learning_starts", "batch_size",
               "tau", "gamma", "train_freq", "gradient_steps", "ent_coef",
               "target_update_interval", "policy_kwargs"}
    return {k: v for k, v in copy.deepcopy(config).items() if k in allowed}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _summary_without_table(result: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in result.items() if k != "ic_sequence"}


def evaluate_model(model: SAC, env: TradingEnv, episodes: int,
                   resampling_minutes: int, roll_window: int,
                   min_periods: int,
                   progress_name: str,
                   batch_size: int = 4096) -> Dict[str, Any]:
    """一次性预测指定校验数据，之后统一计算 IC。"""
    del episodes  # 数据范围由完整环境或固定窗口环境自身定义。
    started = time.monotonic()
    expected_rows = (sum(end - start for _, start, end in env.fixed_windows)
                     if hasattr(env, "fixed_windows") else
                     sum(np.isfinite(env.future_ret_h[pos]).sum()
                         for pos in env._code_positions.values()))
    print(f"[{progress_name}_INFERENCE_START] rows={expected_rows} "
          f"batch_size={batch_size}", flush=True)
    frame = predict_environment(model, env, batch_size=batch_size)
    inference_seconds = time.monotonic() - started
    print(f"[{progress_name}_INFERENCE_END] rows={len(frame)} "
          f"elapsed={inference_seconds:.1f}s", flush=True)
    result = evaluate_predictions(
        frame, resampling_minutes=resampling_minutes,
        roll_window=roll_window, min_periods=min_periods)
    result["rows"] = int(len(frame))
    result["inference_seconds"] = float(inference_seconds)
    result["elapsed_seconds"] = float(time.monotonic() - started)
    return result


class QuickICCallback(BaseCallback):
    """四组轮换的快速验证只监控，不保存模型、不触发早停。"""

    def __init__(self, env, eval_freq, log_path, resampling_minutes,
                 roll_window, min_periods, batch_size):
        super().__init__(verbose=0)
        self.env, self.eval_freq, self.log_path = env, int(eval_freq), log_path
        self.resampling_minutes = int(resampling_minutes)
        self.roll_window = int(roll_window)
        self.min_periods = int(min_periods)
        self.batch_size = int(batch_size)

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq:
            return True
        group = self.env.select_next_group()
        print(f"[QUICK_IC_START] step={self.num_timesteps} group={group} "
              f"windows={len(self.env.fixed_windows)} monitor_only=True", flush=True)
        result = evaluate_model(self.model, self.env, len(self.env.fixed_windows),
                                self.resampling_minutes, self.roll_window,
                                self.min_periods, "QUICK_IC", self.batch_size)
        summary = _summary_without_table(result)
        record = {"train_step": int(self.num_timesteps), "group_id": group, **summary}
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False,
                                    default=_json_default) + "\n")
        score = float(result["selection_score"])
        self.logger.record(f"quick_ic/group_{group}/min_asset_ic", score)
        for code, metrics in result["assets"].items():
            self.logger.record(f"quick_ic/group_{group}/{code}_ic",
                               metrics["rolling_pearson_ic_mean"])
        self.logger.dump(self.num_timesteps)
        print(f"[QUICK_IC_END] step={self.num_timesteps} group={group} "
              f"min_asset_ic={score:.6f} elapsed={result['elapsed_seconds']:.1f}s",
              flush=True)
        return True


class FullICSelectionCallback(BaseCallback):
    """完整校验集按 min(各品种滚动 Pearson IC 均值) 保存最佳模型。"""

    def __init__(self, env, eval_freq, model_dir, log_dir,
                 resampling_minutes, roll_window, min_periods,
                 min_valid_rolling_ic, early_stop_patience_evals,
                 early_stop_min_evals, early_stop_min_delta,
                 early_stop_start_timesteps, batch_size,
                 enable_early_stop=True, verbose=1):
        super().__init__(verbose=verbose)
        self.env, self.eval_freq = env, int(eval_freq)
        self.model_dir, self.log_dir = model_dir, log_dir
        self.resampling_minutes = int(resampling_minutes)
        self.roll_window = int(roll_window)
        self.min_periods = int(min_periods)
        self.min_valid_rolling_ic = int(min_valid_rolling_ic)
        self.batch_size = int(batch_size)
        self.patience = int(early_stop_patience_evals)
        self.min_evals = int(early_stop_min_evals)
        self.min_delta = float(early_stop_min_delta)
        self.start_steps = int(early_stop_start_timesteps)
        self.enable_early_stop = bool(enable_early_stop)
        self.eval_count = 0
        self.monitor_count = 0
        self.no_improve_count = 0
        self.best_score = -np.inf
        self.early_reference = -np.inf
        self.best_model_path = os.path.join(model_dir, "best_model", "best_model")

    def _eligible(self, result):
        score = float(result["selection_score"])
        if not np.isfinite(score) or not result["assets"]:
            return False, "selection_score_nonfinite"
        for code, metrics in result["assets"].items():
            if int(metrics["valid_rolling_ic"]) < self.min_valid_rolling_ic:
                return False, (f"{code}_valid_rolling_ic<"
                               f"{self.min_valid_rolling_ic}")
            if float(metrics["prediction_std"]) <= 1e-12:
                return False, f"{code}_constant_prediction"
        return True, "ok"

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq:
            return True
        self.eval_count += 1
        print(f"[FULL_IC_START] step={self.num_timesteps} eval={self.eval_count}",
              flush=True)
        result = evaluate_model(self.model, self.env, self.env.n_assets,
                                self.resampling_minutes, self.roll_window,
                                self.min_periods, "FULL_IC", self.batch_size)
        score = float(result["selection_score"])
        eligible, reason = self._eligible(result)
        improved = eligible and score > self.best_score
        if improved:
            self.best_score = score
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            self.model.save(self.best_model_path)
            print(f"[BEST_IC_MODEL] step={self.num_timesteps} score={score:.6f} "
                  f"path={self.best_model_path}", flush=True)

        summary = _summary_without_table(result)
        record = {"train_step": int(self.num_timesteps), "eval_count": self.eval_count,
                  "eligible": eligible, "eligibility_reason": reason,
                  "saved_as_best": improved, **summary}
        with open(os.path.join(self.log_dir, "full_ic_validation.jsonl"), "a",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False,
                                    default=_json_default) + "\n")
        sequence = result["ic_sequence"]
        sequence.to_csv(
            os.path.join(self.log_dir,
                         f"full_rolling_ic_step_{self.num_timesteps}.csv"),
            index=False)
        self.logger.record("full_ic/min_asset_ic", score)
        for code, metrics in result["assets"].items():
            self.logger.record(f"full_ic/{code}_ic",
                               metrics["rolling_pearson_ic_mean"])
            self.logger.record(f"full_ic/{code}_mse", metrics["mse_raw_return"])
        self.logger.dump(self.num_timesteps)

        continue_training = True
        if self.enable_early_stop and eligible and self.num_timesteps >= self.start_steps:
            self.monitor_count += 1
            if score > self.early_reference + self.min_delta:
                self.early_reference = score
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
            if self.monitor_count >= self.min_evals and self.no_improve_count >= self.patience:
                continue_training = False
                print(f"[EARLY_STOP_IC] step={self.num_timesteps} "
                      f"no_improve={self.no_improve_count} best={self.best_score:.6f}",
                      flush=True)
        print(f"[FULL_IC_END] step={self.num_timesteps} score={score:.6f} "
              f"eligible={eligible} saved={improved} rows={result['rows']} "
              f"elapsed={result['elapsed_seconds']:.1f}s", flush=True)
        return continue_training


class TrainingProgressCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.last_time, self.last_step = time.monotonic(), 0

    def _on_step(self):
        now = time.monotonic()
        if now - self.last_time >= 15.0:
            print(f"[TRAIN_PROGRESS] step={self.num_timesteps} "
                  f"delta_steps={self.num_timesteps-self.last_step} "
                  f"elapsed={now-self.last_time:.1f}s", flush=True)
            self.last_time, self.last_step = now, self.num_timesteps
        return True


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                features: List[str], env_config: Dict[str, Any],
                sac_config: Dict[str, Any], signal_config=None,
                output_dir: str = "", tensorboard_dir: str = "",
                total_timesteps: int = 100000, eval_freq: int = 10000,
                save_freq: int = 10000, val_window_steps: int = 60,
                val_windows_per_asset: int = 160, full_eval_freq: int = 50000,
                validation_batch_size: int = 4096,
                ic_resampling_minutes: int = 5,
                ic_roll_window: int = 15, ic_min_periods: int = 5,
                min_valid_rolling_ic: int = 20,
                early_stop_patience_evals: int = 3,
                early_stop_min_evals: int = 4,
                early_stop_min_delta: float = 0.0,
                early_stop_start_timesteps: int = 120000,
                enable_early_stop: bool = True, verbose: int = 1,
                **unused) -> Tuple[SAC, Dict[str, Any]]:
    if not output_dir:
        raise ValueError("output_dir 不能为空")
    if int(eval_freq) <= 0 or int(full_eval_freq) <= 0:
        raise ValueError("eval_freq 和 full_eval_freq 必须大于0")
    model_dir, log_dir = os.path.join(output_dir, "models"), os.path.join(output_dir, "logs")
    tensorboard_dir = tensorboard_dir or os.path.join(output_dir, "tensorboard")
    for path in (model_dir, log_dir, tensorboard_dir):
        os.makedirs(path, exist_ok=True)

    train_raw = create_env(train_df, "train", features, env_config, signal_config)
    env_config = dict(env_config,
                      ret_scale_by_code=dict(train_raw.ret_scale_by_code))
    print(f"[TARGET_SCALE] train_std_by_code={env_config['ret_scale_by_code']}")
    fixed_env = FixedWindowValidationEnv(
        _sanitize_dataframe(val_df, features), features,
        {"env_config": dict(env_config, mode="val"),
         "signal_config": signal_config or {}},
        window_steps=val_window_steps, windows_per_asset=val_windows_per_asset)
    full_env = create_env(val_df, "val", features, env_config, signal_config)
    with open(os.path.join(log_dir, "validation_windows.json"), "w",
              encoding="utf-8") as handle:
        json.dump(fixed_env.window_records(), handle, ensure_ascii=False, indent=2)

    config = _sanitize_sac_config(sac_config)
    gamma = float(config.get("gamma", 0.0))
    if abs(gamma) > 1e-12:
        raise ValueError("rl016 的标签已包含完整未来5分钟收益；MSE单步拟合要求 gamma=0.0")
    config["gamma"] = 0.0
    if env_config.get("use_tcn", False):
        policy_kwargs = dict(config.get("policy_kwargs", {}))
        policy_kwargs["features_extractor_class"] = TCNFeaturesExtractor
        policy_kwargs.setdefault("features_extractor_kwargs", {})
        config["policy_kwargs"] = policy_kwargs

    train_env = Monitor(ResetFixWrapper(train_raw),
                        filename=os.path.join(log_dir, "train_monitor.csv"))
    model = SAC("MlpPolicy", train_env, tensorboard_log=tensorboard_dir,
                verbose=verbose, seed=env_config.get("seed"), **config)
    if model.device.type == "cuda":
        th.backends.cudnn.deterministic = False
        th.backends.cudnn.benchmark = True
    print(f"[MODEL_INPUT] observation_space={train_raw.observation_space.shape} "
          f"action_space={train_raw.action_space.shape} use_tcn={env_config.get('use_tcn')} ")
    print(f"[SAC_RUNTIME] device={model.device} gamma={model.gamma} "
          f"batch_size={model.batch_size} train_freq={model.train_freq} "
          f"gradient_steps={model.gradient_steps}", flush=True)

    quick = QuickICCallback(fixed_env, eval_freq,
                            os.path.join(log_dir, "quick_ic_validation.jsonl"),
                            ic_resampling_minutes, ic_roll_window,
                            ic_min_periods,
                            validation_batch_size)
    full = FullICSelectionCallback(
        full_env, full_eval_freq, model_dir, log_dir,
        ic_resampling_minutes, ic_roll_window, ic_min_periods,
        min_valid_rolling_ic, early_stop_patience_evals,
        early_stop_min_evals, early_stop_min_delta,
        early_stop_start_timesteps, validation_batch_size,
        enable_early_stop, verbose)
    callbacks = [quick, full,
                 CheckpointCallback(save_freq=int(save_freq),
                                    save_path=os.path.join(model_dir, "checkpoints"),
                                    name_prefix="sac_model", verbose=verbose),
                 TrainingProgressCallback()]

    config_info = {
        "version": "rl016", "objective": "negative_standardized_mse",
        "selection_metric": "min_asset_rolling_pearson_ic_mean",
        "env_config": env_config, "sac_config": config,
        "signal_config": signal_config or {}, "features": features,
        "validation": {"groups": 4, "window_steps": int(val_window_steps),
                       "windows_per_asset": int(val_windows_per_asset),
                       "quick_eval_freq": int(eval_freq),
                       "full_eval_freq": int(full_eval_freq),
                       "batch_size": int(validation_batch_size),
                       "ic_resampling_minutes": int(ic_resampling_minutes),
                       "ic_roll_window": int(ic_roll_window),
                       "ic_min_periods": int(ic_min_periods),
                       "min_valid_rolling_ic": int(min_valid_rolling_ic)},
        "train_size": int(len(train_df)), "val_size": int(len(val_df)),
        "training_date": datetime.now().isoformat(),
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config_info, handle, ensure_ascii=False, indent=2,
                  default=_json_default)

    print(f"开始 rl016 训练: steps={total_timesteps}, quick={eval_freq}, "
          f"full={full_eval_freq}, selection=min_asset_rolling_IC", flush=True)
    model.learn(total_timesteps=int(total_timesteps), callback=callbacks,
                log_interval=4)
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)

    best_path = full.best_model_path
    if os.path.isfile(best_path) or os.path.isfile(best_path + ".zip"):
        # 最终复核继续使用训练设备，避免重新加载后意外落到CPU。
        review_model = SAC.load(best_path, device=model.device)
        reviewed = "best_ic"
    else:
        review_model, reviewed = model, "final_no_full_eval"
    print(f"[FINAL_REVIEW_MODEL] kind={reviewed} device={review_model.device}",
          flush=True)
    review = evaluate_model(review_model, full_env, full_env.n_assets,
                            ic_resampling_minutes, ic_roll_window,
                            ic_min_periods, "FINAL_IC", validation_batch_size)
    review_summary = {"model": reviewed, **_summary_without_table(review)}
    review_path = os.path.join(log_dir, "final_full_ic_validation.json")
    with open(review_path, "w", encoding="utf-8") as handle:
        json.dump(review_summary, handle, ensure_ascii=False, indent=2,
                  default=_json_default)
    review["ic_sequence"].to_csv(
        os.path.join(log_dir, "final_rolling_ic.csv"), index=False)
    info = {"model_path": final_path, "best_model_path": best_path,
            "config_path": config_path, "log_dir": log_dir,
            "full_validation_path": review_path,
            "best_selection_score": float(full.best_score)}
    print(f"[TRAIN_END] final={final_path} best_ic={best_path} "
          f"best_score={full.best_score:.6f}", flush=True)
    return model, info
