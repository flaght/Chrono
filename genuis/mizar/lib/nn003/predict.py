"""加载 HybridTransformer-MSE 最佳模型并批量评估测试集。"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from .data import SequenceStore
from .inference import (load_checkpoint, model_config_from_checkpoint,
                        predict_store, resolve_device)
from .metrics import evaluate_predictions
from .model import HybridTransformerRegressor


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def predict_test_set(model_path: str, config_path: str,
                     test_df: pd.DataFrame, output_path: str,
                     inference_batch_size: Optional[int] = None,
                     device: str = "auto") -> pd.DataFrame:
    runtime_device = resolve_device(device)
    checkpoint = load_checkpoint(model_path, runtime_device)
    with open(config_path, "r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    features = list(checkpoint["features"])
    missing = {"trade_time", "code", "nxt1_ret", *features} - set(test_df.columns)
    if missing:
        raise ValueError(f"测试集缺少字段: {sorted(missing)}")
    model = HybridTransformerRegressor(
        **model_config_from_checkpoint(checkpoint)).to(runtime_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    store = SequenceStore(
        test_df, features, int(checkpoint["lookback"]),
        ret_scale_by_code=checkpoint["ret_scale_by_code"])
    train_config = dict(run_config.get("train_config", {}))
    batch_size = int(inference_batch_size or
                     train_config.get("inference_batch_size", 4096))
    print(f"[HYBRID_TRANSFORMER_MSE_TEST_INFERENCE] rows={len(test_df)} "
          f"batch_size={batch_size} device={runtime_device}", flush=True)
    records = predict_store(model, store, batch_size, runtime_device)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records.to_csv(output_path, index=False)
    metrics = evaluate_predictions(
        records,
        resampling_minutes=int(train_config.get("ic_resampling_minutes", 5)),
        roll_window=int(train_config.get("ic_roll_window", 15)),
        min_periods=int(train_config.get("ic_min_periods", 5)))
    summary = {key: value for key, value in metrics.items()
               if key != "ic_sequence"}
    stem = os.path.splitext(output_path)[0]
    with open(stem + "_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2,
                  default=_json_default)
    metrics["ic_sequence"].to_csv(stem + "_rolling_ic.csv", index=False)
    print(f"[HYBRID_TRANSFORMER_MSE_TEST_END] "
          f"score={summary['selection_score']:.6f} output={output_path}",
          flush=True)
    return records


__all__ = ["predict_test_set"]
