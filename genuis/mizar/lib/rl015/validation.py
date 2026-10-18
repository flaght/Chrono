"""固定窗口验证：仅使用时间和标签有效性选窗，不根据收益大小选窗。"""
import numpy as np
import time
from lib.rl015.envs import TradingEnv


class FixedWindowValidationEnv(TradingEnv):
    def __init__(self, *args, window_steps=60, windows_per_asset=160, **kwargs):
        super().__init__(*args, **kwargs)
        width, requested = int(window_steps), int(windows_per_asset)
        if width <= 0 or requested <= 0:
            raise ValueError("验证窗口长度和每品种窗口数必须为正整数")
        candidates = {}
        for code, positions in self._code_positions.items():
            windows = []
            valid = np.isfinite(self.future_ret_h[positions])
            for left, right in self._segments_by_code[code]:
                edges = np.diff(np.r_[False, valid[left:right], False].astype(np.int8))
                for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    # 等长、不重叠；不足长度的碎片只在最终完整验证中评分。
                    windows.extend((int(s), int(s + width))
                                   for s in range(left + int(a), left + int(b) - width + 1, width))
            candidates[code] = windows
        count = min(requested, *(len(v) for v in candidates.values()))
        # windows_per_asset 是四组总预算；各组等数量，并至少有两个短窗口。
        count = (count // 4) * 4
        if count < 8:
            raise ValueError("每品种至少需8个完整窗口才能形成4组多窗口验证；请减小窗口长度或增加预算")
        self.window_groups = [[] for _ in range(4)]
        for code, windows in candidates.items():
            # 按时间顺序均匀覆盖整个验证跨度；所有评估轮次复用同一列表。
            indices = np.linspace(0, len(windows) - 1, count, dtype=int)
            # 时间序列交错分配，避免一组只对应连续某一段月份。
            for rank, i in enumerate(indices):
                self.window_groups[rank % 4].append((code, *windows[i]))
        self.group_rng = np.random.default_rng(self.env_config.get('seed', 42))
        self.group_order = []
        self.active_group = 0
        self.fixed_windows = self.window_groups[0]
        self.window_cursor = 0
        self.debug_reset_log = False

    def select_next_group(self):
        if not self.group_order:
            self.group_order = self.group_rng.permutation(4).tolist()
        self.active_group = self.group_order.pop(0)
        self.fixed_windows = self.window_groups[self.active_group]
        self.window_cursor = 0
        return self.active_group + 1

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if self.window_cursor == 0:
            self.progress_started = self.progress_reported = time.monotonic()
            self.progress_rows = 0
        code, start, end = self.fixed_windows[self.window_cursor % len(self.fixed_windows)]
        self.window_cursor += 1
        self.active_code = code
        # 保留原始品种位置，TCN可以读取评分窗口之前的真实历史，不重置为填充。
        self.active_positions = self._code_positions[code]
        self.episode_start_offset = self.current_code_offset = start
        self.episode_end_offset_exclusive = end
        self.current_step = int(self.active_positions[start])
        self.history = []
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.progress_rows += 1
        now = time.monotonic()
        if now - self.progress_reported >= 15:
            width = self.episode_end_offset_exclusive - self.episode_start_offset
            total = len(self.fixed_windows) * width
            speed = self.progress_rows / max(now - self.progress_started, 1e-9)
            print(f"[VAL_WINDOW_PROGRESS] group={self.active_group + 1} code={self.active_code} "
                  f"rows={self.progress_rows}/{total} speed={speed:.1f}/s "
                  f"eta={max(0, total-self.progress_rows)/speed:.1f}s", flush=True)
            self.progress_reported = now
        # Monitor累计后得到每窗口平均每步奖励；各窗口等长且各品种等数量。
        return obs, reward / (self.episode_end_offset_exclusive - self.episode_start_offset), done, info

    def window_records(self):
        return [dict(group_id=group_id + 1, code=code, start_offset=start, end_offset=end,
                     start_time=str(self.df.iloc[self._code_positions[code][start]].trade_time),
                     end_time=str(self.df.iloc[self._code_positions[code][end-1]].trade_time))
                for group_id, windows in enumerate(self.window_groups)
                for code, start, end in windows]
