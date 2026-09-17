"""rl016 的批量、无状态预测。"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def _evaluation_ranges(env) -> List[Tuple[str, str, np.ndarray]]:
    """完整环境返回每个品种；固定窗口环境返回当前组的全部窗口。"""
    if hasattr(env, "fixed_windows"):
        return [
            (code, f"fixed_{index}_{code}_{start}_{end}",
             env._code_positions[code][int(start):int(end)])
            for index, (code, start, end) in enumerate(env.fixed_windows)
        ]
    ranges = []
    for code in env.asset_codes:
        # 保留无效标签行，使 rolling(window=15) 的窗口位置与既有因子评估一致；
        # pandas rolling.corr 会自行忽略窗口中的 NaN 标签。
        positions = env._code_positions[code]
        ranges.append((code, f"full_{code}", positions))
    return ranges


def predict_environment(model, env, batch_size: int = 4096,
                        deterministic: bool = True) -> pd.DataFrame:
    """
    对指定环境做一次完整预测。

    环境状态和下一观测不依赖 action，因此校验/测试无需逐行 env.step；
    内部按 batch_size 分块只用于控制内存，结果等价于逐行确定性推理。
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("validation_batch_size 必须大于0")
    frames = []
    was_training = model.policy.training
    try:
        for code, segment, positions in _evaluation_ranges(env):
            if positions.size == 0:
                continue
            action_parts = []
            for left in range(0, len(positions), batch_size):
                chunk = positions[left:left + batch_size]
                observations = env.get_observation_batch(chunk)
                actions, _ = model.predict(
                    observations, deterministic=bool(deterministic))
                actions = np.asarray(actions, dtype=np.float64).reshape(-1)
                if len(actions) != len(chunk):
                    raise ValueError("批量预测返回的动作数量与输入数量不一致")
                action_parts.append(np.clip(actions, -1.0, 1.0))
            action = np.concatenate(action_parts)
            scale = float(env.ret_scale_by_code[code])
            future = env.future_ret_h[positions].astype(np.float64, copy=False)
            predicted_z = action * env.prediction_bound_std
            target_z = future / scale
            predicted_ret = predicted_z * scale
            error = predicted_z - target_z
            squared_error = error ** 2
            frames.append(pd.DataFrame({
                "trade_time": env.df.iloc[positions]["trade_time"].to_numpy(),
                "code": code,
                "evaluation_segment": segment,
                "label_valid": np.isfinite(future),
                "action_raw": action,
                "prediction_bound_std": env.prediction_bound_std,
                "predicted_z": predicted_z,
                "target_z": target_z,
                "predicted_ret_h": predicted_ret,
                "future_ret_h": future,
                "prediction_error_z": error,
                "squared_error_z": squared_error,
                "ret_scale": scale,
                "reward": -squared_error,
                "reward_scaled": -squared_error * env.reward_scale,
                "direction": np.sign(predicted_ret).astype(np.int8),
                "confidence": np.abs(predicted_z),
                "net_er_out": predicted_z,
                "er_value": predicted_z,
            }))
    finally:
        model.policy.set_training_mode(was_training)
    if not frames:
        raise ValueError("没有可供批量推理的有效校验/测试样本")
    return pd.concat(frames, ignore_index=True)
