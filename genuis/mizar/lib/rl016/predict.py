"""使用 rl016 最佳 IC 模型输出具有收益率单位的预测。"""

import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from kichaos.stable3.sac import SAC

from lib.rl016.envs import TradingEnv
from lib.rl016.inference import predict_environment
from lib.rl016.metrics import evaluate_predictions


def _sanitize(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[features] = out[features].apply(pd.to_numeric, errors="coerce")
    out[features] = out[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


class SignalGenerator:
    def __init__(self, model_path: str, config_path: str,
                 deterministic: bool = True):
        with open(config_path, "r", encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.model = SAC.load(model_path, device="auto")
        print(f"[PREDICT_MODEL] device={self.model.device} path={model_path}",
              flush=True)
        self.features = self.config["features"]
        self.env_config = self.config["env_config"]
        self.signal_config = self.config.get("signal_config", {})
        self.deterministic = bool(deterministic)

    def predict_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        config = {"env_config": dict(self.env_config, mode="test"),
                  "signal_config": self.signal_config}
        env = TradingEnv(_sanitize(df, self.features), self.features, config)
        batch_size = int(
            self.config.get("validation", {}).get("batch_size", 4096))
        result = predict_environment(
            self.model, env, batch_size=batch_size,
            deterministic=self.deterministic)
        columns = ["trade_time", "code", "label_valid", "action_raw",
                   "predicted_z", "target_z", "predicted_ret_h",
                   "future_ret_h", "prediction_error_z", "squared_error_z",
                   "ret_scale", "reward_scaled", "direction", "confidence",
                   "net_er_out", "er_value"]
        return result[columns]


def predict_test_set(model_path: str, config_path: str, test_df: pd.DataFrame,
                     output_path: Optional[str] = None,
                     deterministic: bool = True) -> pd.DataFrame:
    generator = SignalGenerator(model_path, config_path, deterministic)
    result = generator.predict_signals(test_df)
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        result.to_csv(output_path, index=False)
        validation = generator.config.get("validation", {})
        metrics = evaluate_predictions(
            result,
            resampling_minutes=int(
                validation.get("ic_resampling_minutes", 5)),
            roll_window=int(validation.get("ic_roll_window", 15)),
            min_periods=int(validation.get("ic_min_periods", 5)))
        sequence = metrics.pop("ic_sequence")
        metrics_path = os.path.splitext(output_path)[0] + "_metrics.json"
        sequence_path = os.path.splitext(output_path)[0] + "_rolling_ic.csv"
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2,
                      default=lambda x: x.item() if isinstance(x, np.generic) else x)
        sequence.to_csv(sequence_path, index=False)
        print(f"[TEST_END] rows={len(result)} predictions={output_path} "
              f"metrics={metrics_path}", flush=True)
    return result
