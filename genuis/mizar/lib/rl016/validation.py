"""固定四组快速校验；选窗只依赖时间与标签有效性。"""
import time
import numpy as np
from lib.rl016.envs import TradingEnv


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
                edges = np.diff(
                    np.r_[False, valid[left:right], False].astype(np.int8))
                for a, b in zip(np.flatnonzero(edges == 1),
                                np.flatnonzero(edges == -1)):
                    begin, finish = left + int(a), left + int(b)
                    windows.extend((int(s), int(s + width)) for s in range(
                        begin, finish - width + 1, width))
            candidates[code] = windows

        count = min(requested, *(len(v) for v in candidates.values()))
        count = (count // 4) * 4
        if count < 8:
            raise ValueError("每品种至少需8个完整窗口才能形成4组多窗口验证")

        self.window_groups = [[] for _ in range(4)]
        for code, windows in candidates.items():
            indices = np.linspace(0, len(windows) - 1, count, dtype=int)
            for rank, idx in enumerate(indices):
                self.window_groups[rank % 4].append((code, *windows[idx]))
        self.group_rng = np.random.default_rng(self.env_config.get("seed", 42))
        self.group_order = []
        self.active_group = 0
        self.fixed_windows = self.window_groups[0]
        self.window_cursor = 0
        self.debug_reset_log = False

    def select_next_group(self):
        if not self.group_order:
            self.group_order = self.group_rng.permutation(4).tolist()
        self.active_group = int(self.group_order.pop(0))
        self.fixed_windows = self.window_groups[self.active_group]
        self.window_cursor = 0
        return self.active_group + 1

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if self.window_cursor == 0:
            self.progress_started = self.progress_reported = time.monotonic()
            self.progress_rows = 0
        code, start, end = self.fixed_windows[
            self.window_cursor % len(self.fixed_windows)]
        self.window_cursor += 1
        self.active_code = code
        self.active_positions = self._code_positions[code]
        self.episode_start_offset = self.current_code_offset = int(start)
        self.episode_end_offset_exclusive = int(end)
        self.current_step = int(self.active_positions[start])
        self.history = []
        return self._get_obs()

    def step(self, action):
        result = super().step(action)
        self.progress_rows += 1
        now = time.monotonic()
        if now - self.progress_reported >= 15.0:
            width = self.episode_end_offset_exclusive - self.episode_start_offset
            total = len(self.fixed_windows) * width
            speed = self.progress_rows / max(now - self.progress_started, 1e-9)
            print(f"[VAL_WINDOW_PROGRESS] group={self.active_group + 1} "
                  f"rows={self.progress_rows}/{total} speed={speed:.1f}/s "
                  f"eta={max(0, total-self.progress_rows)/max(speed, 1e-9):.1f}s",
                  flush=True)
            self.progress_reported = now
        return result

    def window_records(self):
        return [
            dict(group_id=group_id + 1, code=code, start_offset=start,
                 end_offset=end,
                 start_time=str(self.df.iloc[self._code_positions[code][start]].trade_time),
                 end_time=str(self.df.iloc[self._code_positions[code][end - 1]].trade_time))
            for group_id, windows in enumerate(self.window_groups)
            for code, start, end in windows
        ]
