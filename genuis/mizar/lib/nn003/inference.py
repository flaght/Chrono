"""HybridTransformer-MSE 批量推理。"""

from typing import Dict

import numpy as np
import pandas as pd
import torch as th

from .data import SequenceStore


def resolve_device(requested: str = "auto") -> th.device:
    requested = str(requested or "auto").lower()
    if requested == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not th.cuda.is_available():
        raise RuntimeError("参数要求 CUDA，但 torch.cuda.is_available() 为 False")
    return th.device(requested)


def load_checkpoint(path: str, device: th.device) -> Dict:
    try:
        return th.load(path, map_location=device, weights_only=False)
    except TypeError:
        return th.load(path, map_location=device)


def predict_store(model: th.nn.Module, store: SequenceStore,
                  batch_size: int, device: th.device) -> pd.DataFrame:
    if int(batch_size) <= 0:
        raise ValueError("inference_batch_size 必须大于0")
    model.eval()
    frames = []
    with th.no_grad():
        for code in store.asset_codes:
            positions = store.code_positions[code]
            predictions = []
            for left in range(0, len(positions), int(batch_size)):
                chunk = positions[left:left + int(batch_size)]
                observations = th.from_numpy(
                    store.observation_batch(chunk)).to(device)
                predictions.append(
                    model(observations).detach().cpu().numpy().astype(np.float64))
            predicted_z = np.concatenate(predictions)
            scale = float(store.ret_scale_by_code[code])
            future = store.future_ret_h[positions].astype(np.float64, copy=False)
            predicted_ret = predicted_z * scale
            target_z = future / scale
            error = predicted_z - target_z
            frames.append(pd.DataFrame({
                "trade_time": store.trade_times[positions], "code": code,
                "evaluation_segment": f"full_{code}",
                "label_valid": np.isfinite(future),
                "action_raw": predicted_z / model.prediction_bound_std,
                "prediction_bound_std": model.prediction_bound_std,
                "predicted_z": predicted_z, "target_z": target_z,
                "predicted_ret_h": predicted_ret, "future_ret_h": future,
                "prediction_error_z": error, "squared_error_z": error ** 2,
                "ret_scale": scale,
                "direction": np.sign(predicted_ret).astype(np.int8),
                "confidence": np.abs(predicted_z),
                "net_er_out": predicted_z, "er_value": predicted_z,
            }))
    if not frames:
        raise ValueError("没有可供推理的数据")
    return pd.concat(frames, ignore_index=True)


def model_config_from_checkpoint(checkpoint: Dict) -> Dict:
    required = {"n_features", "lookback", "d_model", "n_heads", "e_layers",
                "d_layers", "d_ff", "dropout", "activation",
                "prediction_bound_std"}
    config = dict(checkpoint.get("model_config", {}))
    missing = required - set(config)
    if missing:
        raise ValueError(f"模型检查点缺少配置: {sorted(missing)}")
    return config


__all__ = ["load_checkpoint", "predict_store", "resolve_device",
           "model_config_from_checkpoint"]
