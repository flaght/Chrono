"""HybridTransformer-MSE 的分钟级多资产因果时序批次。"""

from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd


class SequenceStore:
    """按需构造窗口，避免预展开全部 (lookback, feature) 数据。"""

    def __init__(self, frame: pd.DataFrame, features: List[str], lookback: int,
                 ret_scale_by_code: Optional[Dict[str, float]] = None,
                 fit_return_scale: bool = False):
        required = {"trade_time", "code", "nxt1_ret", *features}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"SequenceStore 数据缺少字段: {sorted(missing)}")
        if frame.empty or int(lookback) <= 0:
            raise ValueError("数据不能为空且 lookback 必须大于0")
        self.features, self.lookback = list(features), int(lookback)
        self.df = frame.copy()
        self.df["trade_time"] = pd.to_datetime(
            self.df["trade_time"], errors="raise")
        self.df["code"] = self.df["code"].astype(str)
        self.df = self.df.sort_values(
            ["code", "trade_time"]).reset_index(drop=True)
        if self.df.duplicated(["code", "trade_time"]).any():
            raise ValueError("数据存在重复的 code/trade_time")
        values = self.df[self.features].apply(
            pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self.feature_values = np.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0)
        self.future_ret_h = pd.to_numeric(
            self.df["nxt1_ret"], errors="coerce").to_numpy(dtype=np.float64)
        self.future_ret_h[~np.isfinite(self.future_ret_h)] = np.nan
        self.trade_times = self.df["trade_time"].to_numpy()
        self.codes = self.df["code"].to_numpy(dtype=str)
        self.asset_codes = [str(code) for code in pd.unique(self.codes)]
        self.code_positions = {
            code: np.flatnonzero(self.codes == code).astype(np.int64)
            for code in self.asset_codes}
        self.position_in_code = np.empty(len(self.df), dtype=np.int64)
        self.segment_start_in_code = np.empty(len(self.df), dtype=np.int64)
        for code, positions in self.code_positions.items():
            self.position_in_code[positions] = np.arange(len(positions))
            gaps = (self.df.iloc[positions]["trade_time"].diff()
                    .dt.total_seconds().to_numpy())
            if np.any(gaps[1:] <= 0):
                raise ValueError(f"{code} 存在重复或逆序时间")
            starts = [0] + (np.flatnonzero(gaps[1:] != 60.0) + 1).tolist()
            boundaries = starts + [len(positions)]
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                self.segment_start_in_code[positions[left:right]] = int(left)

        configured = dict(ret_scale_by_code or {})
        self.ret_scale_by_code: Dict[str, float] = {}
        for code, positions in self.code_positions.items():
            valid = self.future_ret_h[positions]
            valid = valid[np.isfinite(valid)]
            if fit_return_scale:
                scale = float(np.std(valid)) if len(valid) else np.nan
            elif code in configured:
                scale = float(configured[code])
            else:
                raise ValueError(f"缺少 {code} 的训练集收益标准差")
            if not np.isfinite(scale) or scale <= 1e-8:
                raise ValueError(f"{code} 的训练收益标准差无效: {scale}")
            self.ret_scale_by_code[code] = scale
        self.valid_positions_by_code = {
            code: positions[np.isfinite(self.future_ret_h[positions])]
            for code, positions in self.code_positions.items()}
        for code, positions in self.valid_positions_by_code.items():
            if not len(positions):
                raise ValueError(f"{code} 没有有效标签")

    def observation_batch(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        result = np.empty(
            (len(positions), self.lookback, len(self.features)),
            dtype=np.float32)
        lags = np.arange(self.lookback - 1, -1, -1, dtype=np.int64)
        selected_codes = self.codes[positions]
        for code in pd.unique(selected_codes):
            rows = np.flatnonzero(selected_codes == code)
            selected = positions[rows]
            offsets = self.position_in_code[selected]
            starts = self.segment_start_in_code[selected]
            history_offsets = np.maximum(
                offsets[:, None] - lags[None, :], starts[:, None])
            history_positions = self.code_positions[str(code)][history_offsets]
            result[rows] = self.feature_values[history_positions]
        return result

    def target_z_batch(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        target = np.empty(len(positions), dtype=np.float32)
        selected_codes = self.codes[positions]
        for code in pd.unique(selected_codes):
            rows = np.flatnonzero(selected_codes == code)
            target[rows] = (self.future_ret_h[positions[rows]] /
                            self.ret_scale_by_code[str(code)])
        return target


def _take_cyclic(permutation: np.ndarray, cursor: int, count: int,
                 rng: np.random.Generator):
    parts, remaining = [], int(count)
    while remaining > 0:
        take = min(remaining, len(permutation) - cursor)
        parts.append(permutation[cursor:cursor + take])
        cursor, remaining = cursor + take, remaining - take
        if cursor >= len(permutation):
            permutation, cursor = rng.permutation(permutation), 0
    return np.concatenate(parts), permutation, cursor


def balanced_batches(store: SequenceStore, batch_size: int, epoch: int,
                     seed: int = 42,
                     samples_per_epoch: Optional[int] = None
                     ) -> Iterator[np.ndarray]:
    """每批次让各品种贡献尽可能相同的样本数。"""
    codes = list(store.asset_codes)
    if len(codes) < 2 or int(batch_size) < len(codes):
        raise ValueError("至少需要两个品种，且 batch_size 不能小于品种数")
    if samples_per_epoch is None or int(samples_per_epoch) <= 0:
        samples_per_epoch = len(codes) * max(
            len(store.valid_positions_by_code[code]) for code in codes)
    rng = np.random.default_rng(int(seed) + int(epoch) * 1009)
    permutations = {code: rng.permutation(store.valid_positions_by_code[code])
                    for code in codes}
    cursors = {code: 0 for code in codes}
    produced = 0
    while produced < int(samples_per_epoch):
        width = min(int(batch_size), int(samples_per_epoch) - produced)
        base, extra = divmod(width, len(codes))
        parts = []
        for index, code in enumerate(codes):
            chosen, permutations[code], cursors[code] = _take_cyclic(
                permutations[code], cursors[code],
                base + (1 if index < extra else 0), rng)
            parts.append(chosen)
        batch = np.concatenate(parts)
        rng.shuffle(batch)
        produced += len(batch)
        yield batch


__all__ = ["SequenceStore", "balanced_batches"]
