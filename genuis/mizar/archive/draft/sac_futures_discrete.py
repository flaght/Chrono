"""
SAC Futures Discrete - 基于SAC算法的期货单标的离散动作强化学习实现

功能:
- 离散方向决策：Actor 输出 3 维 Logits，通过 Softmax 映射到 [观望, 做多, 做空] 概率
- 信号权重：做多概率 - 做空概率，作为连续信号（保留分类意图，又保留置信度大小）
- 奖励：信号权重 * 未来 N 分钟累计收益 (single_horizon 模式)
- 训练窗口：支持滑动窗口采样（full/half/holding三种stride）
- 不依赖任何自定义库，仅使用标准 kichaos/PyTorch/numpy/pandas/gym
"""
import pdb
import os
import json
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from gym import spaces

from kichaos.stable3.sac import SAC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. TradingEnv - 交易环境
# =============================================================================

class TradingEnv(gym.Env):
    """
    基于预期收益率合成的离散分类强化学习环境。

    动作空间：Box(3,) - 3维 Logits，分别代表 [观望, 做多, 做空] 的意图得分。
    观测空间：纯因子特征向量。

    核心逻辑：
    1. SAC 输出 3D Logits [-1, 1]，环境对其放大后做 Softmax 得到三类概率。
    2. 若做多/做空概率均不超过观望概率，则信号为 0（不开仓）。
    3. 否则信号 = 做多概率 - 做空概率（正数=做多，负数=做空）。
    4. 奖励 = 信号 * 未来 holding_period 分钟累计收益（single_horizon 模式）。

    mode="train"：按照 train_scheme 滑动窗口采样训练片段。
    mode="val"/"infer"：从头到尾评估完整序列。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        config: Dict[str, Any],
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = features

        env_config = config.get("env_config", {})

        self.holding_period       = int(env_config["holding_period"])
        self.reward_scale         = float(env_config.get("reward_scale", 10000.0))
        self.softmax_temperature  = float(env_config.get("softmax_temperature", 5.0))
        self.mode                 = str(env_config.get("mode", "infer")).strip().lower()
        self.max_episode_steps    = int(env_config.get("max_episode_steps", 0))
        self.train_scheme         = str(env_config.get("train_scheme", "full")).strip().lower()

        self.last_open_step = len(self.df) - self.holding_period - 1
        if self.last_open_step < 0:
            raise ValueError(
                f"数据长度不足以支持完整持有 {self.holding_period} 期: len(df)={len(self.df)}"
            )

        # 动作/观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.features),),
            dtype=np.float32
        )

        # 状态
        self.current_step = 0
        self.history: List[Dict] = []
        self.future_ret_h = self._build_future_horizon_returns()

        # 训练窗口
        self.random_start_min_step = 0
        max_valid_start = self.last_open_step
        if self.max_episode_steps > 0:
            max_valid_start = max(0, self.last_open_step - self.max_episode_steps + 1)
        self.random_start_max_step = max_valid_start

        self.train_window_starts: List[int] = []
        self.train_window_order: List[int] = []
        self.train_window_cursor = 0
        self.episode_end_step_exclusive = self.last_open_step + 1
        self.reset_count = 0

        self.seed(seed=env_config.get("seed", 42))

        if self.mode == "train":
            stride_map = {
                "full":    max(1, self.max_episode_steps),
                "half":    max(1, self.max_episode_steps // 2),
                "holding": max(1, self.holding_period),
            }
            stride = stride_map.get(self.train_scheme, max(1, self.max_episode_steps))
            starts = list(range(self.random_start_min_step, self.random_start_max_step + 1, stride))
            if not starts:
                starts = [self.random_start_min_step]
            if starts[-1] != self.random_start_max_step:
                starts.append(self.random_start_max_step)
            self.train_window_starts = starts
            self._reset_train_window_order()

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _build_future_horizon_returns(self) -> np.ndarray:
        ret = pd.to_numeric(self.df.get("nxt1_ret", 0.0), errors="coerce").astype(float).to_numpy()
        ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(ret)
        out = np.full(n, np.nan, dtype=np.float64)
        if self.holding_period <= 0 or n < self.holding_period:
            return out
        valid = np.convolve(ret, np.ones(self.holding_period, dtype=np.float64), mode="valid")
        out[: len(valid)] = valid
        return out

    def _reset_train_window_order(self):
        n = len(self.train_window_starts)
        self.train_window_order = list(range(n))
        if n > 1:
            self.np_random.shuffle(self.train_window_order)
        self.train_window_cursor = 0

    def _next_train_window_start(self) -> int:
        if self.train_window_cursor >= len(self.train_window_order):
            self._reset_train_window_order()
        idx = self.train_window_order[self.train_window_cursor]
        self.train_window_cursor += 1
        return int(self.train_window_starts[idx])

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if self.mode == "train":
            self.current_step = self._next_train_window_start()
            self.episode_end_step_exclusive = min(
                self.last_open_step + 1,
                self.current_step + max(1, self.max_episode_steps)
            ) if self.max_episode_steps > 0 else self.last_open_step + 1
        else:
            self.current_step = 0
            self.episode_end_step_exclusive = self.last_open_step + 1
        self.reset_count += 1
        self.history = []
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        obs = pd.to_numeric(row[self.features], errors='coerce').values.astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def step(self, action: np.ndarray):
        raw_action = action.astype(float)
        if not np.isfinite(raw_action).all():
            raw_action = np.zeros(3, dtype=float)

        # Softmax：放大 logits 突破 Tanh [-1,1] 约束，使概率可达 >90%
        scaled = raw_action * self.softmax_temperature
        exp_a = np.exp(scaled - np.max(scaled))
        probs = exp_a / np.sum(exp_a)  # [p_neutral, p_long, p_short]

        # 信号：若观望概率最大则不开仓，否则 er = p_long - p_short
        if probs[0] > probs[1] and probs[0] > probs[2]:
            er_value = 0.0
            confidence = float(probs[0])
        else:
            er_value = float(probs[1]) - float(probs[2])  # 正=多，负=空
            confidence = float(abs(er_value))

        can_open = self.current_step <= self.last_open_step
        opened   = can_open and er_value != 0.0
        net_er   = er_value if opened else 0.0
        future_ret_h = float(self.future_ret_h[self.current_step]) if can_open else np.nan
        nxt1_ret = float(self.df.iloc[self.current_step].get('nxt1_ret', 0.0))

        target_ret = future_ret_h if np.isfinite(future_ret_h) else 0.0
        step_reward = net_er * target_ret
        if not np.isfinite(step_reward):
            step_reward = 0.0
        scaled_reward = step_reward * self.reward_scale

        trade_time = self.df.iloc[self.current_step].get('trade_time', self.current_step)
        self.history.append({
            'trade_time':    trade_time,
            'raw_action':    f"[{raw_action[0]:.4f},{raw_action[1]:.4f},{raw_action[2]:.4f}]",
            'soft_action':   f"[{probs[0]:.4f},{probs[1]:.4f},{probs[2]:.4f}]",
            'signal':        er_value,
            'er_value':      er_value,
            'net_er_out':    net_er,
            'direction':     1 if er_value > 0 else (-1 if er_value < 0 else 0),
            'confidence':    confidence,
            'opened':        opened,
            'current_ret':   nxt1_ret,
            'future_ret_h':  future_ret_h,
            'target_ret':    target_ret,
            'reward':        step_reward,
            'reward_scaled': scaled_reward,
        })

        self.current_step = min(self.current_step + 1, len(self.df) - 1)
        done = self.current_step >= self.episode_end_step_exclusive
        return self._get_obs(), scaled_reward, done, {}


# =============================================================================
# 2. 训练函数
# =============================================================================

def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    env_config: Dict[str, Any],
    sac_config: Dict[str, Any],
    output_dir: str,
    total_timesteps: int,
    eval_freq: int = 10000,
    eval_n_episodes: int = 1,
    save_freq: int = 50000,
    enable_early_stop: bool = True,
    early_stop_patience_evals: int = 6,
    early_stop_min_evals: int = 6,
    early_stop_min_delta: float = 0.0,
    verbose: int = 1,
) -> Tuple[Any, Dict[str, Any]]:
    """使用 kichaos.stable3 SAC 训练期货离散方向预测模型。"""
    from kichaos.stable3.common.monitor import Monitor
    from kichaos.stable3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

    model_dir = os.path.join(output_dir, "models")
    log_dir   = os.path.join(output_dir, "logs")
    tb_dir    = os.path.join(output_dir, "tensorboard")
    for d in [model_dir, log_dir, tb_dir]:
        os.makedirs(d, exist_ok=True)

    def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in features + ["nxt1_ret"] if c in df.columns]
        df = df.copy()
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    def _make_env(df: pd.DataFrame, mode: str) -> gym.Env:
        cfg = dict(env_config)
        cfg["mode"] = mode
        env = TradingEnv(df=_sanitize(df), features=features, config={"env_config": cfg})
        # 兼容 reset() 可能返回 (obs, info) 的 gym 版本
        class _FixReset(gym.Wrapper):
            def reset(self, **kw):
                r = self.env.reset(**kw)
                return r[0] if isinstance(r, tuple) and len(r) == 2 else r
        return _FixReset(env)

    train_env = Monitor(_make_env(train_df, "train"), filename=os.path.join(log_dir, "train_monitor.csv"))
    val_env   = Monitor(_make_env(val_df, "val"),   filename=os.path.join(log_dir, "val_monitor.csv"))

    allowed_sac_keys = {
        "learning_rate", "buffer_size", "learning_starts", "batch_size",
        "tau", "gamma", "train_freq", "gradient_steps", "ent_coef",
        "target_update_interval", "policy_kwargs",
    }
    clean_sac = {k: v for k, v in sac_config.items() if k in allowed_sac_keys}

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        tensorboard_log=tb_dir,
        verbose=verbose,
        seed=env_config.get("seed"),
        **clean_sac,
    )

    eval_cb = EvalCallback(
        val_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        n_eval_episodes=int(eval_n_episodes),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=verbose,
    )

    callbacks = [eval_cb, CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="sac_model",
        verbose=verbose,
    )]

    if enable_early_stop:
        class _EarlyStop(BaseCallback):
            def __init__(self, cb, patience, min_evals, min_delta, v):
                super().__init__(v)
                self._cb = cb; self._patience = patience
                self._min_evals = min_evals; self._min_delta = min_delta
                self._best = -np.inf; self._cnt = 0; self._prev_n = 0
            def _on_step(self):
                n = len(getattr(self._cb, "evaluations_results", []))
                if n <= self._prev_n:
                    return True
                best = float(getattr(self._cb, "best_mean_reward", -np.inf))
                if best > self._best + self._min_delta:
                    self._best = best; self._cnt = 0
                else:
                    self._cnt += n - self._prev_n
                self._prev_n = n
                if n >= self._min_evals and self._cnt >= self._patience:
                    if self.verbose > 0:
                        print(f"[EARLY STOP] 连续 {self._cnt} 次无提升，停止训练。")
                    return False
                return True
        callbacks.append(_EarlyStop(eval_cb, early_stop_patience_evals, early_stop_min_evals, early_stop_min_delta, verbose))

    logger.info(f"开始训练，总步数: {total_timesteps}, 训练集: {len(train_df)}, 验证集: {len(val_df)}")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, log_interval=4)

    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    logger.info(f"最终模型保存至: {final_path}")

    config_info = {
        "env_config": env_config,
        "sac_config": clean_sac,
        "features": features,
        "total_timesteps": total_timesteps,
        "training_date": datetime.now().isoformat(),
        "train_size": len(train_df),
        "val_size": len(val_df),
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    return model, {
        "model_path": final_path,
        "best_model_path": os.path.join(model_dir, "best_model", "best_model"),
        "config_path": config_path,
    }


# =============================================================================
# 3. 预测函数
# =============================================================================

def predict_test_set(
    model_path: str,
    config_path: str,
    test_df: pd.DataFrame,
    output_path: Optional[str] = None,
    deterministic: bool = True,
) -> pd.DataFrame:
    """加载已训练模型，对测试集逐步预测，返回含信号的 DataFrame。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    features   = cfg["features"]
    env_config = dict(cfg["env_config"])
    env_config["mode"] = "infer"

    def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in features + ["nxt1_ret"] if c in df.columns]
        df = df.copy()
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    env = TradingEnv(df=_sanitize(test_df), features=features, config={"env_config": env_config})
    model = SAC.load(model_path)

    obs = env.reset()
    results = []
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward_scaled, done, _ = env.step(action)
        h = env.history[-1] if env.history else {}
        results.append({
            "trade_time":    h.get("trade_time", ""),
            "raw_action":    h.get("raw_action", ""),
            "soft_action":   h.get("soft_action", ""),
            "er_value":      float(h.get("er_value", 0.0)),
            "net_er_out":    float(h.get("net_er_out", 0.0)),
            "confidence":    float(h.get("confidence", 0.0)),
            "direction":     int(h.get("direction", 0)),
            "current_ret":   float(h.get("current_ret", 0.0)),
            "future_ret_h":  float(h.get("future_ret_h", 0.0)),
            "reward_scaled": float(reward_scaled),
        })
        if done:
            break

    signals_df = pd.DataFrame(results)
    logger.info(f"预测完成，共 {len(signals_df)} 条记录")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        signals_df.to_csv(output_path, index=False)
        logger.info(f"预测结果保存至: {output_path}")

    # 打印简单统计
    if len(signals_df) > 0:
        ic = signals_df["er_value"].corr(signals_df["future_ret_h"], method="spearman")
        logger.info(f"Rank IC (er_value vs future_ret_h): {ic:.6f}")

    return signals_df


# =============================================================================
# 4. 示例入口
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("SAC Futures Discrete - 示例运行")
    logger.info("=" * 60)

    # 示例：真实使用时替换为 pd.read_feather(...)
    train_data = pd.read_feather("./train_data.feather")
    val_data   = pd.read_feather("./val_data.feather")
    test_data  = pd.read_feather("./test_data.feather")
    pdb.set_trace()
    features   = [ "MSUM(120,MDEMA(90,MCPS(90,'high')))", "MDEMA(120,MCPS(120,WMA(90,'twap')))",  
                  "MMAX(120,MDEMA(60,MCPS(90,'low')))",  "MMASSI(120,MPRO(60,MVHF(10,'money')),MAPOSITIVE(10,'twap'))",
                    "MDEMA(120,MCPS(120,MADecay(60,'twap')))",  
                    "MT3(120,MCPS(30,'close'))",  "MA(60,RSI(120,MCPS(120,MA(60,'twap'))))", 
                    "MT3(120,MCPS(60,'high'))","DELTA(90,MMIN(15,MHMA(90,DELTA(90,'close'))))/MDIFF(90,'close')",  
                    "MCPS(120,MT3(90,MMaxDiff(120,'twap')))",
                    "MADecay(5,MMASSI(120,MT3(5,'corr_vwap_bid_size_0'),'twap'))",  
                    "MMeanRes(120,'corr_money_bid_size_0','smart_tick_in_pct')",
                    "WMA(30,MMedian(90,'smart_tick_in_pct'))", 
                    "MMAX(15,MDPO(240,EMA(90,'smart_money_in_pct')))",
                    "RSI(120,MCPS(120,EMA(120,'close')))",
                    "MSUM(120,MDEMA(90,MCPS(90,'low')))",  
                    "MDIFF(90,MMeanRes(120,'corr_money_bid_size_0','smart_tick_in_pct'))",
                    "MSUM(5,MADecay(10,MMedian(90,'smart_tick_in_pct')))",  
                    "MMedian(90,MADecay(10,MT3(5,'smart_tick_in_pct')))"
    ]
    
    
    ## 此处未直接使用15分钟收益率，而是使用1分钟收益率，在env 进行累加。 做一个验证。
    train_data = train_data[['trade_time','code','nxt1_ret_1h'] + features].rename(columns={'nxt1_ret_1h':'nxt1_ret'})
    val_data = val_data[['trade_time','code','nxt1_ret_1h'] + features].rename(columns={'nxt1_ret_1h':'nxt1_ret'})
    test_data = test_data[['trade_time','code','nxt1_ret_1h'] + features].rename(columns={'nxt1_ret_1h':'nxt1_ret'})
    env_config = {
        "holding_period":     15,       # 持仓周期（分钟）
        "reward_scale":       10000.0,  # 奖励放大系数
        "softmax_temperature": 5.0,     # Softmax 温度系数（放大 Logits）
        "mode":               "train",  # train / val / infer
        "max_episode_steps":  500,      # 每个 episode 最大步数（0=不限）
        "train_scheme":       "full",   # 训练窗口 stride 方案：full/half/holding
        "seed":               42,
    }

    sac_config = {
        "learning_rate":          3e-4,
        "buffer_size":            100000,
        "learning_starts":        1000,
        "batch_size":             256,
        "tau":                    0.005,
        "gamma":                  0.97,
        "train_freq":             1,
        "gradient_steps":         1,
        "ent_coef":               "auto",
        "target_update_interval": 1
    }

    output_dir = "./sac_futures_output"

    model, training_info = train_model(
        train_df=train_data, val_df=val_data, features=features,
        env_config=env_config, sac_config=sac_config,
        output_dir=output_dir, total_timesteps=100000, verbose=1,
    )

    signals_df = predict_test_set(
        model_path=training_info["best_model_path"],
        config_path=training_info["config_path"],
        test_df=test_data,
        output_path=os.path.join(output_dir, "predictions.csv"),
    )
