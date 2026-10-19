"""单品种逐 bar 的多 seed HybridTransformer 预测；不依赖 pandas 或标签。"""
from __future__ import annotations


import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import torch as th

from .model import HybridTransformerRegressor


MODEL_VERSION = "hybrid_transformer_loss_v1"


def seed_model_dirs(release_path: str) -> dict[int, str]:
    """定位 release_path/<seed>/<run_id>；多个 run_id 时拒绝猜测。"""
    root = Path(release_path)
    if not root.is_dir():
        raise FileNotFoundError(f"发布目录不存在: {root}")
    run_dirs = {}
    for seed_dir in sorted(root.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.isdigit():
            continue
        candidates = sorted(path for path in seed_dir.iterdir() if path.is_dir())
        if len(candidates) != 1:
            raise ValueError(
                f"{seed_dir} 下应恰好有一个 run_id 目录，实际有 {len(candidates)} 个；"
                "请显式传入 run_dirs")
        run_dirs[int(seed_dir.name)] = str(candidates[0].resolve())
    if len(run_dirs) < 2:
        raise ValueError("等权融合至少需要两个 seed 模型")
    return run_dirs


def _device(requested: str) -> th.device:
    requested = str(requested or "auto").lower()
    if requested == "auto":
        requested = "cuda" if th.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not th.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但 torch.cuda.is_available() 为 False")
    return th.device(requested)


def _checkpoint(path: Path) -> dict:
    # 只加载自己训练并发布的可信 checkpoint；旧版 PyTorch 不支持 weights_only。
    try:
        return th.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return th.load(path, map_location="cpu")


def _bar_time(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError("trade_time 必须为 datetime 或 ISO 格式字符串")


class LiveEnsemblePredictor:
    """一实例对应一个品种；窗口构造与训练用 SequenceStore 保持一致。"""

    def __init__(self, run_dirs: Mapping[int, str], code: str,
                 requested: str = "auto", min_history: int = 1):
        if len(run_dirs) < 2:
            raise ValueError("等权融合至少需要两个 seed 模型")
        self.code = str(code).strip().upper()
        self.device = _device(requested)
        self.models = {}
        self.run_ids = {}
        reference = None
        for seed, run_dir in sorted(run_dirs.items()):
            seed = int(seed)
            path = Path(run_dir)
            with (path / "config.json").open(encoding="utf-8") as handle:
                config = json.load(handle)
            if int(config["env_config"]["seed"]) != seed:
                raise ValueError(f"{path} 的 seed 与目录映射不一致")
            checkpoint = _checkpoint(path / "models" / "best_model.pt")
            if checkpoint.get("version") != MODEL_VERSION:
                raise ValueError(f"{path} 的模型版本不匹配")
            for field in ("features", "lookback", "ret_scale_by_code",
                          "loss_config"):
                if checkpoint[field] != config[field]:
                    raise ValueError(f"{path} 的 checkpoint 与 config.json 的 {field} 不一致")
            architecture = dict(checkpoint["model_config"])
            if architecture != config["architecture"]:
                raise ValueError(f"{path} 的网络结构与 config.json 不一致")
            if architecture["n_features"] != len(config["features"]):
                raise ValueError(f"{path} 的特征数量与网络输入维度不一致")
            if architecture["lookback"] != config["lookback"]:
                raise ValueError(f"{path} 的 lookback 与网络结构不一致")
            signature = {key: config[key] for key in
                         ("features", "lookback", "ret_scale_by_code",
                          "architecture", "loss_config")}
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError(f"{path} 与其他 seed 的模型配置不一致")
            model = HybridTransformerRegressor(**architecture)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.models[seed] = model.to(self.device).eval()
            self.run_ids[seed] = path.name
            del checkpoint

        self.features = list(reference["features"])
        self.lookback = int(reference["lookback"])
        scales = reference["ret_scale_by_code"]
        if self.code not in scales:
            raise ValueError(f"发布模型没有 {self.code} 的训练集收益标准差")
        self.ret_scale = float(scales[self.code])
        if not np.isfinite(self.ret_scale) or self.ret_scale <= 1e-8:
            raise ValueError(f"{self.code} 的训练集收益标准差无效")
        self.min_history = int(min_history)
        if not 1 <= self.min_history <= self.lookback:
            raise ValueError(f"min_history 必须在 1 到 {self.lookback} 之间")
        self.reset()

    def keys(self):
        """兼容原脚本中 models.keys() 的查看方式。"""
        return self.models.keys()

    def reset(self) -> None:
        """新交易流开始时手动清空历史。"""
        self.history = deque(maxlen=self.lookback)
        self.last_time = None
        self.last_segment_reset = False

    def push_values(self, trade_time, code: str, values) -> int:
        """只接收当前及历史 bar 的特征，不接收未来收益。"""
        if str(code).strip().upper() != self.code:
            raise ValueError(f"预测器只支持 {self.code}，收到 {code}")
        timestamp = _bar_time(trade_time)
        if len(values) != len(self.features):
            raise ValueError(f"需要 {len(self.features)} 个特征，收到 {len(values)} 个")
        sanitized = []
        for value in values:
            try:
                sanitized.append(float(value))
            except (TypeError, ValueError):
                sanitized.append(float("nan"))
        vector = np.nan_to_num(np.asarray(sanitized, dtype=np.float32),
                               nan=0.0, posinf=0.0, neginf=0.0)
        self.last_segment_reset = self.last_time is None
        if self.last_time is not None:
            gap = (timestamp - self.last_time).total_seconds()
            if gap <= 0:
                raise ValueError("bar 时间必须严格递增，不能重复或倒序")
            if gap != 60.0:
                # 离线 SequenceStore 在任何非连续一分钟处重新开始序列。
                self.history.clear()
                self.last_segment_reset = True
        self.history.append(vector)
        self.last_time = timestamp
        return len(self.history)

    def push_bar(self, bar: Mapping) -> int:
        missing = [name for name in self.features if name not in bar]
        if missing:
            raise KeyError(f"bar 缺少模型特征: {missing}")
        return self.push_values(bar["trade_time"], bar.get("code", self.code),
                                [bar[name] for name in self.features])

    def warmup(self, bars) -> None:
        """按时间顺序灌入历史 bar，不产生预测，也不需要标签列。"""
        for bar in bars:
            self.push_bar(bar)

    def _window(self) -> np.ndarray:
        """不足 lookback 时，用当前连续片段第一根 bar 左侧填充。"""
        if not self.history:
            raise ValueError("至少需要一根 bar 才能预测")
        observed = np.stack(tuple(self.history), axis=0)
        missing = self.lookback - len(observed)
        if missing:
            observed = np.concatenate(
                (np.repeat(observed[:1], missing, axis=0), observed), axis=0)
        return observed[None, :, :]

    def _predict_current(self):
        count = len(self.history)
        if count < self.min_history:
            return None
        x = th.from_numpy(self._window()).to(self.device)
        # torch 1.7 尚无 inference_mode；no_grad 在推理时同样关闭梯度记录。
        inference_context = getattr(th, "inference_mode", th.no_grad)
        with inference_context():
            per_seed = {seed: float(model(x).item())
                        for seed, model in self.models.items()}
        if not all(np.isfinite(value) for value in per_seed.values()):
            raise FloatingPointError("模型输出 NaN/Inf，已阻止本 bar 产生预测")
        predicted_z = float(np.mean(list(per_seed.values())))
        return {
            "trade_time": self.last_time,
            "code": self.code,
            "predicted_z": predicted_z,
            "predicted_ret_h": predicted_z * self.ret_scale,
            "per_seed_predicted_z": per_seed,
            "history_bars": count,
            "left_padded": count < self.lookback,
            "segment_reset": self.last_segment_reset,
        }

    def predict_values(self, trade_time, code: str, values):
        """高频接入：按 self.features 顺序传入当前 bar 的特征值。"""
        self.push_values(trade_time, code, values)
        return self._predict_current()

    def predict_bar(self, bar: Mapping):
        """常规接入：bar 含 trade_time、可选 code 和全部特征。"""
        self.push_bar(bar)
        return self._predict_current()


def load_model(run_dirs, requested: str = "auto", code: str = "P",
               min_history: int = 1) -> LiveEnsemblePredictor:
    return LiveEnsemblePredictor(run_dirs, code, requested, min_history)
