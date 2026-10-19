"""将逐 bar 的融合 predicted_z 转为已固化的目标信号；不产生订单。"""

from __future__ import annotations

import math
from typing import Mapping


# 与 z07_strategies/6.4.1.transformer_signal_backtest.py 中选入测试集的
# 三组参数一致。只接收 roll_num=0 的无状态信号，不在实盘中重新寻优。
SIGNAL_PRESETS = {
    "1101": ("linear_signal", {"roll_num": 0, "threshold": 0.05}),
    "1103": ("linear_signal", {"roll_num": 0, "threshold": 0.15}),
    "1201": ("linear_threshold_signal", {
        "roll_num": 0, "threshold": 0.01, "upper": 0.075,
    }),
    "1301": ("threshold_signal", {"roll_num": 0, "threshold": 0.02}),
}


class LiveSignalConverter:
    """单值版本的信号函数；输入必须是融合模型返回的 predicted_z。"""

    def __init__(self, signal_id: str):
        self.signal_id = str(signal_id)
        if self.signal_id not in SIGNAL_PRESETS:
            raise ValueError(f"未知信号编号 {signal_id}；可选 {sorted(SIGNAL_PRESETS)}")
        self.signal_method, preset = SIGNAL_PRESETS[self.signal_id]
        self.params = dict(preset)
        if self.params["roll_num"] != 0:
            raise ValueError("逐 bar 转换只支持 roll_num=0")

    def value(self, predicted_z: float) -> float | int:
        """严格复现对应 Lumina 函数的有限输入分支。"""
        z = float(predicted_z)
        if not math.isfinite(z):
            raise FloatingPointError("predicted_z 非有限值，禁止产生交易信号")
        threshold = float(self.params["threshold"])
        if self.signal_method == "linear_signal":
            return max(-1.0, min(1.0, z / threshold))
        if self.signal_method == "linear_threshold_signal":
            upper = float(self.params["upper"])
            magnitude = max(0.0, min(1.0, (abs(z) - threshold) /
                                     (upper - threshold)))
            return math.copysign(magnitude, z) if magnitude else 0.0
        if self.signal_method == "threshold_signal":
            return int(z >= threshold) - int(z <= -threshold)
        raise RuntimeError(f"未实现信号函数: {self.signal_method}")

    def convert(self, prediction: Mapping | None) -> dict | None:
        """返回带 signal 的新记录，不修改预测器原始输出。"""
        if prediction is None:
            return None
        if "predicted_z" not in prediction:
            raise KeyError("预测结果缺少 predicted_z")
        result = dict(prediction)
        result["signal"] = self.value(result["predicted_z"])
        result["signal_method"] = self.signal_method
        result["signal_id"] = self.signal_id
        return result


def generate_live_signals(predictions, signal_id: str):
    """回放预测完成后，批量生成独立的信号结果；不修改预测 DataFrame。"""
    import pandas as pd

    required = {"trade_time", "code", "predicted_z"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"预测结果缺少字段: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("预测结果为空，无法生成信号")
    if {"signal", "signal_id", "signal_method"} & set(predictions.columns):
        raise ValueError("输入已经包含信号字段；请传入原始预测结果")

    converter = LiveSignalConverter(signal_id)
    signals = predictions.copy()
    signals["signal"] = signals["predicted_z"].map(converter.value)
    signals["signal_method"] = converter.signal_method
    signals["signal_id"] = converter.signal_id
    # 与研究评估的采样口径相同，但此列并非下单许可。
    timestamps = pd.to_datetime(signals["trade_time"], errors="raise")
    signals["eval_5m_mark"] = timestamps.dt.minute % 5 == 0
    return signals
