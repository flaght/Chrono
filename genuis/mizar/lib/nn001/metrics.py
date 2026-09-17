"""与现有因子评估及 rl016 一致的重采样滚动 IC 指标。"""

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return np.nan
    x, y = x[valid], y[valid]
    if float(x.std(ddof=0)) <= 1e-12 or float(y.std(ddof=0)) <= 1e-12:
        return np.nan
    if method == "spearman":
        x, y = x.rank(method="average"), y.rank(method="average")
    result = x.corr(y, method="pearson")
    return float(result) if pd.notna(result) else np.nan


def _finite_mean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def evaluate_predictions(records: pd.DataFrame,
                         resampling_minutes: int = 5,
                         roll_window: int = 15,
                         min_periods: int = 5) -> Dict[str, Any]:
    """
    评估口径：minute % 5 == 0 后，对未来收益和 predicted_z 计算
    rolling(window=15, min_periods=5).corr。
    """
    required = {"trade_time", "code", "predicted_z", "predicted_ret_h",
                "future_ret_h"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"IC 评估缺少字段: {sorted(missing)}")
    if int(resampling_minutes) <= 0 or int(roll_window) <= 0:
        raise ValueError("resampling_minutes 和 roll_window 必须大于0")
    if int(min_periods) <= 0 or int(min_periods) > int(roll_window):
        raise ValueError("min_periods 必须在 [1, roll_window] 内")

    frame = records.copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="raise")
    if "evaluation_segment" not in frame.columns:
        frame["evaluation_segment"] = frame["code"].astype(str)
    frame = frame.sort_values(
        ["code", "evaluation_segment", "trade_time"]).reset_index(drop=True)

    assets: Dict[str, Dict[str, Any]] = {}
    sequence_parts = []
    for code, asset in frame.groupby("code", sort=True):
        code_sequences = []
        for segment, part in asset.groupby("evaluation_segment", sort=False):
            part = part.sort_values("trade_time")
            sampled = part[
                part["trade_time"].dt.minute.mod(resampling_minutes).eq(0)
            ].copy()
            if sampled.empty:
                continue
            sampled["rolling_pearson_ic"] = (
                pd.to_numeric(sampled["future_ret_h"], errors="coerce")
                .rolling(window=roll_window, min_periods=min_periods)
                .corr(pd.to_numeric(sampled["predicted_z"], errors="coerce"))
            )
            sampled["evaluation_segment"] = str(segment)
            sampled["code"] = str(code)
            code_sequences.append(sampled)
            sequence_parts.append(sampled[[
                "trade_time", "code", "evaluation_segment", "predicted_z",
                "future_ret_h", "rolling_pearson_ic"
            ]])

        sampled_asset = (pd.concat(code_sequences, ignore_index=True)
                         if code_sequences else pd.DataFrame())
        valid_raw = asset[["predicted_ret_h", "future_ret_h"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        pred = valid_raw["predicted_ret_h"].to_numpy(dtype=np.float64)
        actual = valid_raw["future_ret_h"].to_numpy(dtype=np.float64)
        error = pred - actual
        pred_var = float(np.var(pred)) if pred.size else np.nan
        if pred.size and pred_var > 1e-20:
            slope = float(np.cov(pred, actual, ddof=0)[0, 1] / pred_var)
            intercept = float(np.mean(actual) - slope * np.mean(pred))
        else:
            slope, intercept = np.nan, np.nan
        zero_mse = float(np.mean(actual ** 2)) if actual.size else np.nan
        model_mse = float(np.mean(error ** 2)) if error.size else np.nan

        if sampled_asset.empty:
            rolling_ic = pd.Series(dtype=float)
            sampled_pred = pd.Series(dtype=float)
            sampled_ret = pd.Series(dtype=float)
        else:
            rolling_ic = sampled_asset["rolling_pearson_ic"]
            sampled_pred = sampled_asset["predicted_z"]
            sampled_ret = sampled_asset["future_ret_h"]
        assets[str(code)] = {
            "rows": int(len(asset)),
            "valid_label_rows": int(len(valid_raw)),
            "sampled_rows": int(len(sampled_asset)),
            "total_pearson_ic": _safe_corr(sampled_pred, sampled_ret),
            "total_spearman_ic": _safe_corr(
                sampled_pred, sampled_ret, method="spearman"),
            "rolling_pearson_ic_mean": _finite_mean(rolling_ic),
            "rolling_pearson_ic_std": (
                float(rolling_ic.std(ddof=1))
                if int(rolling_ic.notna().sum()) > 1 else np.nan),
            "rolling_pearson_positive_rate": (
                float((rolling_ic.dropna() > 0).mean())
                if rolling_ic.notna().any() else np.nan),
            "valid_rolling_ic": int(rolling_ic.notna().sum()),
            "mse_raw_return": model_mse,
            "mae_raw_return": (float(np.mean(np.abs(error)))
                               if error.size else np.nan),
            "zero_prediction_mse": zero_mse,
            "mse_skill_vs_zero": (
                float(1.0 - model_mse / zero_mse)
                if np.isfinite(zero_mse) and zero_mse > 0 else np.nan),
            "prediction_mean": float(np.mean(pred)) if pred.size else np.nan,
            "prediction_std": float(np.std(pred)) if pred.size else np.nan,
            "actual_mean": float(np.mean(actual)) if actual.size else np.nan,
            "actual_std": float(np.std(actual)) if actual.size else np.nan,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }

    scores = [metrics["rolling_pearson_ic_mean"]
              for metrics in assets.values()]
    finite_scores = [score for score in scores if np.isfinite(score)]
    selection_score = (float(min(finite_scores))
                       if assets and len(finite_scores) == len(assets)
                       else np.nan)
    ic_sequence = (pd.concat(sequence_parts, ignore_index=True)
                   if sequence_parts else pd.DataFrame())
    return {
        "selection_metric": "min_asset_rolling_pearson_ic_mean",
        "selection_score": selection_score,
        "resampling_minutes": int(resampling_minutes),
        "roll_window": int(roll_window),
        "min_periods": int(min_periods),
        "assets": assets,
        "ic_sequence": ic_sequence,
    }


__all__ = ["evaluate_predictions"]
