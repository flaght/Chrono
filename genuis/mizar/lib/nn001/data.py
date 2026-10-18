"""分钟级多资产因果时序批次构造。"""

from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd


class SequenceStore:
    """
    保存原始二维特征，仅在取批次时构造 TCN 窗口。

    这样不会把约 135 万条样本预展开为 (20, 24)，可显著降低内存占用。
    窗口不会跨品种，也不会跨非连续的一分钟时间断点；断点开头不足
    lookback 的位置复制该连续段第一行，与 rl016 环境一致。
    """

    def __init__(self, frame: pd.DataFrame, features: List[str], lookback: int,
                 ret_scale_by_code: Optional[Dict[str, float]] = None,
                 fit_return_scale: bool = False):
        required = {"trade_time", "code", "nxt1_ret", *features}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"SequenceStore 数据缺少字段: {sorted(missing)}")
        if frame.empty:
            raise ValueError("SequenceStore 数据不能为空")
        if int(lookback) <= 0:
            raise ValueError("lookback 必须大于0")

        self.features = list(features)
        self.lookback = int(lookback)
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
            for code in self.asset_codes
        }
        self.position_in_code = np.empty(len(self.df), dtype=np.int64)
        self.segment_start_in_code = np.empty(len(self.df), dtype=np.int64)
        for code, positions in self.code_positions.items():
            self.position_in_code[positions] = np.arange(len(positions))
            times = self.df.iloc[positions]["trade_time"]
            gaps = times.diff().dt.total_seconds().to_numpy()
            if np.any(gaps[1:] <= 0):
                raise ValueError(f"{code} 存在重复或逆序时间")
            starts = [0] + (np.flatnonzero(gaps[1:] != 60.0) + 1).tolist()
            boundaries = starts + [len(positions)]
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                self.segment_start_in_code[positions[left:right]] = int(left)

        configured = dict(ret_scale_by_code or {})
        self.ret_scale_by_code: Dict[str, float] = {}
        for code, positions in self.code_positions.items():
            valid_values = self.future_ret_h[positions]
            valid_values = valid_values[np.isfinite(valid_values)]
            if fit_return_scale:
                scale = float(np.std(valid_values)) if len(valid_values) else np.nan
            elif code in configured:
                scale = float(configured[code])
            else:
                raise ValueError(f"缺少 {code} 的训练集收益标准差")
            if not np.isfinite(scale) or scale <= 1e-8:
                raise ValueError(f"{code} 的训练收益标准差无效: {scale}")
            self.ret_scale_by_code[code] = scale

        self.valid_positions_by_code = {
            code: positions[np.isfinite(self.future_ret_h[positions])]
            for code, positions in self.code_positions.items()
        }
        for code, positions in self.valid_positions_by_code.items():
            if len(positions) == 0:
                raise ValueError(f"{code} 没有有效标签")

    def observation_batch(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        result = np.empty(
            (len(positions), self.lookback, len(self.features)),
            dtype=np.float32)
        lags = np.arange(self.lookback - 1, -1, -1, dtype=np.int64)
        for code in pd.unique(self.codes[positions]):
            rows = np.flatnonzero(self.codes[positions] == code)
            selected = positions[rows]
            offsets = self.position_in_code[selected]
            starts = self.segment_start_in_code[selected]
            history_offsets = offsets[:, None] - lags[None, :]
            history_offsets = np.maximum(history_offsets, starts[:, None])
            history_positions = self.code_positions[str(code)][history_offsets]
            result[rows] = self.feature_values[history_positions]
        return result

    def target_z_batch(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        target = np.empty(len(positions), dtype=np.float32)
        for code in pd.unique(self.codes[positions]):
            rows = np.flatnonzero(self.codes[positions] == code)
            target[rows] = (self.future_ret_h[positions[rows]] /
                            self.ret_scale_by_code[str(code)])
        return target


def _take_cyclic(permutation: np.ndarray, cursor: int, count: int,
                 rng: np.random.Generator):
    """从一个品种循环无放回取样，不足时重新打乱后继续。"""
    parts = []
    remaining = int(count)
    while remaining > 0:
        available = len(permutation) - cursor
        take = min(remaining, available)
        parts.append(permutation[cursor:cursor + take])
        cursor += take
        remaining -= take
        if cursor >= len(permutation):
            permutation = rng.permutation(permutation)
            cursor = 0
    return np.concatenate(parts), permutation, cursor


def balanced_batches(store: SequenceStore, batch_size: int, epoch: int,
                     seed: int = 42,
                     samples_per_epoch: Optional[int] = None
                     ) -> Iterator[np.ndarray]:
    """每个批次尽量包含相同数量的 RB、HC 样本。"""
    codes = list(store.asset_codes)
    if len(codes) < 2:
        raise ValueError("多资产基线至少需要两个品种")
    if int(batch_size) < len(codes):
        raise ValueError("batch_size 不能小于品种数")
    if samples_per_epoch is None or int(samples_per_epoch) <= 0:
        samples_per_epoch = len(codes) * max(
            len(store.valid_positions_by_code[c]) for c in codes)
    samples_per_epoch = int(samples_per_epoch)
    rng = np.random.default_rng(int(seed) + int(epoch) * 1009)
    permutations = {
        code: rng.permutation(store.valid_positions_by_code[code])
        for code in codes
    }
    cursors = {code: 0 for code in codes}
    produced = 0
    batch_index = 0
    while produced < samples_per_epoch:
        width = min(int(batch_size), samples_per_epoch - produced)
        base, extra = divmod(width, len(codes))
        parts = []
        for index, code in enumerate(codes):
            count = base + (1 if index < extra else 0)
            chosen, permutations[code], cursors[code] = _take_cyclic(
                permutations[code], cursors[code], count, rng)
            parts.append(chosen)
        batch = np.concatenate(parts)
        rng.shuffle(batch)
        yield batch
        produced += len(batch)
        batch_index += 1


__all__ = ["SequenceStore", "balanced_batches"]
