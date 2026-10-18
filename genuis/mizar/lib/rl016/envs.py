"""rl016 自包含的多资产连续收益预测环境。"""

import random
from typing import Any, Dict, List

import gym
import numpy as np
import pandas as pd
from gym import spaces


class TradingEnv(gym.Env):
    """
    一分钟特征预测预计算的未来五分钟累计收益。

    多资产约束：episode 不跨品种；训练每轮每品种各取一个窗口；TCN
    回看不跨品种和分钟断点；各品种使用训练集收益标准差归一化。

        predicted_z = action * prediction_bound_std
        target_z = future_ret_h / train_std(code)
        reward = -reward_scale * (predicted_z - target_z) ** 2
    """

    metadata = {"render.modes": []}

    def __init__(self, df: pd.DataFrame, features: List[str],
                 config: Dict[str, Any]):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.features = list(features)
        self.config = dict(config)
        self.env_config = dict(config.get("env_config", {}))
        self.signal_config = dict(config.get("signal_config", {}) or {})
        required = {"trade_time", "code", "nxt1_ret", *self.features}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"环境数据缺少字段: {sorted(missing)}")
        if self.df.empty:
            raise ValueError("环境数据不能为空")

        self.mode = str(self.env_config["mode"]).strip().lower()
        if self.mode not in {"train", "val", "test"}:
            raise ValueError("mode 必须是 train、val 或 test")
        self.holding_period = int(self.env_config["holding_period"])
        self.reward_scale = float(self.env_config.get("reward_scale", 1.0))
        self.prediction_bound_std = float(
            self.env_config.get("prediction_bound_std", 3.0))
        self.use_tcn = bool(self.env_config.get("use_tcn", False))
        self.lookback = (max(1, int(self.env_config["default_lookback"]))
                         if self.use_tcn else 1)
        self.max_episode_steps = max(
            0, int(self.env_config.get("max_episode_steps", 0)))
        self.train_scheme = str(
            self.env_config.get("train_scheme", "half")).strip().lower()
        self.asset_sampling = str(
            self.env_config.get("asset_sampling", "balanced")).strip().lower()
        if not np.isfinite(self.prediction_bound_std) or self.prediction_bound_std <= 0:
            raise ValueError("prediction_bound_std 必须是正数")
        if not np.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale 必须是正数")
        if self.train_scheme not in {"full", "half", "holding"}:
            raise ValueError("train_scheme 必须为 full、half 或 holding")
        if self.asset_sampling != "balanced":
            raise ValueError("rl016 当前只支持 asset_sampling='balanced'")

        self.df["trade_time"] = pd.to_datetime(
            self.df["trade_time"], errors="raise")
        self._feature_values = self.df[self.features].apply(
            pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self._feature_values = np.nan_to_num(
            self._feature_values, nan=0.0, posinf=0.0, neginf=0.0)
        self.future_ret_h = pd.to_numeric(
            self.df["nxt1_ret"], errors="coerce").to_numpy(dtype=np.float64)
        self.future_ret_h[~np.isfinite(self.future_ret_h)] = np.nan

        self._codes = self.df["code"].astype(str).to_numpy()
        self.asset_codes = [str(code) for code in pd.unique(self._codes)]
        self.n_assets = len(self.asset_codes)
        self._code_positions = {
            code: np.flatnonzero(self._codes == code)
            for code in self.asset_codes
        }
        self._position_in_code = np.empty(len(self.df), dtype=np.int64)
        for positions in self._code_positions.values():
            self._position_in_code[positions] = np.arange(len(positions))

        self._segment_start_in_code = np.zeros(len(self.df), dtype=np.int64)
        self._segments_by_code: Dict[str, List[tuple]] = {}
        self._build_segments()
        self.ret_scale_by_code = self._build_return_scales()
        self.train_windows_by_code = self._build_train_windows_by_code()

        obs_shape = ((self.lookback, len(self.features))
                     if self.use_tcn else (len(self.features), ))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)

        self.current_step = 0
        self.current_code_offset = 0
        self.episode_start_offset = 0
        self.episode_end_offset_exclusive = 1
        self.active_code = self.asset_codes[0]
        self.active_positions = self._code_positions[self.active_code]
        self.eval_asset_cursor = 0
        self.history: List[Dict[str, Any]] = []
        self.reset_count = 0
        self.debug_reset_log = True
        self.debug_reset_log_every = 5
        self.seed(int(self.env_config.get("seed", 42)))

        self.train_window_orders: Dict[str, List[int]] = {}
        self.train_window_cursors: Dict[str, int] = {}
        self.train_asset_order: List[str] = []
        self.train_asset_cursor = 0
        if self.mode == "train":
            self._reset_train_sampling()
            print("[ENV_INIT][train] scheme={0} asset_sampling={1} "
                  "max_episode_steps={2} assets={3} windows={4}".format(
                      self.train_scheme, self.asset_sampling,
                      self.max_episode_steps, ",".join(self.asset_codes),
                      {k: len(v) for k, v in self.train_windows_by_code.items()}),
                  flush=True)
        else:
            print(f"[ENV_INIT][{self.mode}] asset_episodes="
                  f"{','.join(self.asset_codes)}", flush=True)

    def _build_segments(self):
        """在每个品种内部按严格连续的一分钟划分时段。"""
        for code, positions in self._code_positions.items():
            times = self.df.iloc[positions]["trade_time"]
            gaps = times.diff().dt.total_seconds().to_numpy()
            if np.any(gaps[1:] <= 0):
                raise ValueError(f"{code} 存在重复或逆序时间")
            starts = [0] + (np.flatnonzero(gaps[1:] != 60.0) + 1).tolist()
            boundaries = starts + [len(positions)]
            segments = []
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                self._segment_start_in_code[positions[left:right]] = left
                segments.append((int(left), int(right)))
            self._segments_by_code[code] = segments

    def _build_return_scales(self) -> Dict[str, float]:
        """训练集估计尺度；val/test 必须显式收到训练尺度。"""
        configured = self.env_config.get("ret_scale_by_code")
        scales = {}
        for code, positions in self._code_positions.items():
            if isinstance(configured, dict) and code in configured:
                scale = float(configured[code])
            elif self.mode == "train":
                values = self.future_ret_h[positions]
                values = values[np.isfinite(values)]
                scale = float(np.std(values)) if len(values) else 0.0
            else:
                raise ValueError(
                    f"验证/测试缺少 {code} 的训练尺度 ret_scale_by_code")
            if not np.isfinite(scale) or scale <= 1e-8:
                raise ValueError(f"{code} 的训练收益标准差无效: {scale}")
            scales[code] = scale
        return scales

    def _build_train_windows_by_code(self) -> Dict[str, List[tuple]]:
        windows_by_code = {}
        for code, positions in self._code_positions.items():
            valid = np.isfinite(self.future_ret_h[positions])
            windows = []
            for seg_left, seg_right in self._segments_by_code[code]:
                mask = valid[seg_left:seg_right]
                edges = np.diff(np.r_[False, mask, False].astype(np.int8))
                for left, right in zip(np.flatnonzero(edges == 1),
                                       np.flatnonzero(edges == -1)):
                    left, right = int(left + seg_left), int(right + seg_left)
                    length = right - left
                    width = (min(self.max_episode_steps, length)
                             if self.max_episode_steps > 0 else length)
                    if width <= 0:
                        continue
                    if self.train_scheme == "half":
                        stride = max(1, width // 2)
                    elif self.train_scheme == "holding":
                        stride = max(1, min(self.holding_period, width))
                    else:
                        stride = width
                    for start in range(left, right, stride):
                        end = min(start + width, right)
                        windows.append((int(start), int(end)))
                        if end == right:
                            break
            if not windows:
                raise ValueError(f"{code} 没有可训练的有效标签窗口")
            windows_by_code[code] = windows
        return windows_by_code

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _reshuffle_asset_order(self):
        self.train_asset_order = list(self.asset_codes)
        if len(self.train_asset_order) > 1:
            self.np_random.shuffle(self.train_asset_order)
        self.train_asset_cursor = 0

    def _reset_train_sampling(self):
        for code, windows in self.train_windows_by_code.items():
            order = list(range(len(windows)))
            if len(order) > 1:
                self.np_random.shuffle(order)
            self.train_window_orders[code] = order
            self.train_window_cursors[code] = 0
        self._reshuffle_asset_order()

    def _next_train_episode(self):
        if self.train_asset_cursor >= len(self.train_asset_order):
            self._reshuffle_asset_order()
        code = self.train_asset_order[self.train_asset_cursor]
        self.train_asset_cursor += 1
        order = self.train_window_orders[code]
        cursor = self.train_window_cursors[code]
        if cursor >= len(order):
            order = list(range(len(self.train_windows_by_code[code])))
            if len(order) > 1:
                self.np_random.shuffle(order)
            self.train_window_orders[code] = order
            cursor = 0
        start, end = self.train_windows_by_code[code][order[cursor]]
        self.train_window_cursors[code] = cursor + 1
        return code, start, end

    def _get_obs(self):
        if not self.use_tcn:
            return self._feature_values[self.current_step].astype(
                np.float32, copy=False)
        code = self._codes[self.current_step]
        positions = self._code_positions[code]
        offset = int(self._position_in_code[self.current_step])
        seg_start = int(self._segment_start_in_code[self.current_step])
        start = max(seg_start, offset - self.lookback + 1)
        obs = self._feature_values[positions[start:offset + 1]]
        if len(obs) < self.lookback:
            obs = np.concatenate([
                np.repeat(obs[:1], self.lookback - len(obs), axis=0), obs
            ], axis=0)
        return np.nan_to_num(obs.astype(np.float32, copy=False),
                             nan=0.0, posinf=0.0, neginf=0.0)

    def get_observation_batch(self, positions: np.ndarray) -> np.ndarray:
        """批量构造观测；每个位置仍只读取同品种、同连续分钟段的历史。"""
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        if positions.size == 0:
            return np.empty((0, ) + self.observation_space.shape,
                            dtype=np.float32)
        codes = self._codes[positions]
        if np.any(codes != codes[0]):
            raise ValueError("一个推理批次只能包含一个品种")
        if not self.use_tcn:
            return self._feature_values[positions].astype(np.float32, copy=False)

        code = str(codes[0])
        code_positions = self._code_positions[code]
        offsets = self._position_in_code[positions]
        segment_starts = self._segment_start_in_code[positions]
        lags = np.arange(self.lookback - 1, -1, -1, dtype=np.int64)
        history_offsets = offsets[:, None] - lags[None, :]
        # 时段开头不足 lookback 时复制本时段第一行，与 _get_obs 一致。
        history_offsets = np.maximum(history_offsets, segment_starts[:, None])
        history_positions = code_positions[history_offsets]
        observations = self._feature_values[history_positions]
        return np.nan_to_num(observations.astype(np.float32, copy=False),
                             nan=0.0, posinf=0.0, neginf=0.0)

    def _log_reset_window(self):
        if not self.debug_reset_log or self.reset_count % self.debug_reset_log_every:
            return
        start_idx = int(self.active_positions[self.episode_start_offset])
        end_idx = int(self.active_positions[self.episode_end_offset_exclusive - 1])
        print(f"[ENV_RESET][{self.mode}] reset={self.reset_count} "
              f"code={self.active_code} len="
              f"{self.episode_end_offset_exclusive-self.episode_start_offset} "
              f"time=[{self.df.iloc[start_idx]['trade_time']} -> "
              f"{self.df.iloc[end_idx]['trade_time']}]", flush=True)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if self.mode == "train":
            code, start, end = self._next_train_episode()
            positions = self._code_positions[code]
        else:
            code = self.asset_codes[self.eval_asset_cursor % self.n_assets]
            self.eval_asset_cursor += 1
            raw = self._code_positions[code]
            positions = raw[np.isfinite(self.future_ret_h[raw])]
            if not len(positions):
                raise ValueError(f"{code} 没有有效校验/测试标签")
            start, end = 0, len(positions)
        self.active_code = code
        self.active_positions = positions
        self.episode_start_offset = self.current_code_offset = int(start)
        self.episode_end_offset_exclusive = int(end)
        self.current_step = int(self.active_positions[self.current_code_offset])
        self.history = []
        self.reset_count += 1
        self._log_reset_window()
        return self._get_obs()

    def step(self, action: np.ndarray):
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        if raw.size != 1:
            raise ValueError(f"rl016 需要一维动作，实际 shape={np.shape(action)}")
        action_value = float(raw[0]) if np.isfinite(raw[0]) else 0.0
        action_value = float(np.clip(action_value, -1.0, 1.0))
        future_ret_h = float(self.future_ret_h[self.current_step])
        if not np.isfinite(future_ret_h):
            raise ValueError("当前样本无有效未来标签")

        ret_scale = float(self.ret_scale_by_code[self.active_code])
        target_z = future_ret_h / ret_scale
        predicted_z = action_value * self.prediction_bound_std
        predicted_ret_h = predicted_z * ret_scale
        error_z = predicted_z - target_z
        squared_error = error_z ** 2
        reward_scaled = -squared_error * self.reward_scale
        direction = 1 if predicted_ret_h > 0 else (-1 if predicted_ret_h < 0 else 0)
        self.history.append({
            "trade_time": self.df.iloc[self.current_step]["trade_time"],
            "code": self.active_code,
            "label_valid": True,
            "action_raw": action_value,
            "prediction_bound_std": self.prediction_bound_std,
            "predicted_z": predicted_z,
            "target_z": target_z,
            "predicted_ret_h": predicted_ret_h,
            "future_ret_h": future_ret_h,
            "prediction_error_z": error_z,
            "squared_error_z": squared_error,
            "ret_scale": ret_scale,
            "reward": -squared_error,
            "reward_scaled": reward_scaled,
            "direction": direction,
            "confidence": abs(predicted_z),
            "net_er_out": predicted_z,
            "er_value": predicted_z,
        })
        next_offset = self.current_code_offset + 1
        done = next_offset >= self.episode_end_offset_exclusive
        if not done:
            self.current_code_offset = next_offset
            self.current_step = int(self.active_positions[next_offset])
        return self._get_obs(), float(reward_scaled), bool(done), {}

    def render(self, mode="human"):
        return None

    def close(self):
        return None
