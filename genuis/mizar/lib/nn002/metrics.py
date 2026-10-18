"""与现有因子评估一致的5分钟重采样滚动 IC 指标。"""

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


def _safe_corr(x, y, method="pearson"):
    x, y = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return np.nan
    x, y = x[valid], y[valid]
    if float(x.std(ddof=0)) <= 1e-12 or float(y.std(ddof=0)) <= 1e-12:
        return np.nan
    if method == "spearman":
        x, y = x.rank(method="average"), y.rank(method="average")
    value = x.corr(y)
    return float(value) if pd.notna(value) else np.nan


def _finite_mean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def evaluate_predictions(records: pd.DataFrame, resampling_minutes: int = 5,
                         roll_window: int = 15,
                         min_periods: int = 5) -> Dict[str, Any]:
    required = {"trade_time", "code", "predicted_z", "predicted_ret_h",
                "future_ret_h"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"IC 评估缺少字段: {sorted(missing)}")
    if resampling_minutes <= 0 or roll_window <= 0:
        raise ValueError("resampling_minutes 和 roll_window 必须大于0")
    if min_periods <= 0 or min_periods > roll_window:
        raise ValueError("min_periods 必须在 [1, roll_window] 内")
    frame = records.copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="raise")
    if "evaluation_segment" not in frame:
        frame["evaluation_segment"] = frame["code"].astype(str)
    frame = frame.sort_values(
        ["code", "evaluation_segment", "trade_time"]).reset_index(drop=True)
    assets, sequence_parts = {}, []
    for code, asset in frame.groupby("code", sort=True):
        pieces = []
        for segment, part in asset.groupby("evaluation_segment", sort=False):
            sampled = part.sort_values("trade_time")
            sampled = sampled[
                sampled.trade_time.dt.minute.mod(resampling_minutes).eq(0)
            ].copy()
            if sampled.empty:
                continue
            sampled["rolling_pearson_ic"] = (
                pd.to_numeric(sampled.future_ret_h, errors="coerce")
                .rolling(roll_window, min_periods=min_periods)
                .corr(pd.to_numeric(sampled.predicted_z, errors="coerce")))
            sampled["evaluation_segment"], sampled["code"] = str(segment), str(code)
            pieces.append(sampled)
            sequence_parts.append(sampled[[
                "trade_time", "code", "evaluation_segment", "predicted_z",
                "future_ret_h", "rolling_pearson_ic"]])
        sampled_asset = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        valid = asset[["predicted_ret_h", "future_ret_h"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        pred = valid.predicted_ret_h.to_numpy(dtype=np.float64)
        actual = valid.future_ret_h.to_numpy(dtype=np.float64)
        error = pred - actual
        pred_var = float(np.var(pred)) if pred.size else np.nan
        if pred.size and pred_var > 1e-20:
            slope = float(np.cov(pred, actual, ddof=0)[0, 1] / pred_var)
            intercept = float(np.mean(actual) - slope * np.mean(pred))
        else:
            slope, intercept = np.nan, np.nan
        zero_mse = float(np.mean(actual ** 2)) if actual.size else np.nan
        model_mse = float(np.mean(error ** 2)) if error.size else np.nan
        rolling = (sampled_asset.rolling_pearson_ic if not sampled_asset.empty
                   else pd.Series(dtype=float))
        sampled_pred = (sampled_asset.predicted_z if not sampled_asset.empty
                        else pd.Series(dtype=float))
        sampled_ret = (sampled_asset.future_ret_h if not sampled_asset.empty
                       else pd.Series(dtype=float))
        assets[str(code)] = {
            "rows": int(len(asset)), "valid_label_rows": int(len(valid)),
            "sampled_rows": int(len(sampled_asset)),
            "total_pearson_ic": _safe_corr(sampled_pred, sampled_ret),
            "total_spearman_ic": _safe_corr(sampled_pred, sampled_ret, "spearman"),
            "rolling_pearson_ic_mean": _finite_mean(rolling),
            "rolling_pearson_ic_std": (
                float(rolling.std(ddof=1))
                if rolling.notna().sum() > 1 else np.nan),
            "rolling_pearson_positive_rate": float((rolling.dropna() > 0).mean()) if rolling.notna().any() else np.nan,
            "valid_rolling_ic": int(rolling.notna().sum()),
            "mse_raw_return": model_mse,
            "mae_raw_return": float(np.mean(np.abs(error))) if error.size else np.nan,
            "zero_prediction_mse": zero_mse,
            "mse_skill_vs_zero": float(1 - model_mse / zero_mse) if zero_mse > 0 else np.nan,
            "prediction_mean": float(np.mean(pred)) if pred.size else np.nan,
            "prediction_std": float(np.std(pred)) if pred.size else np.nan,
            "actual_mean": float(np.mean(actual)) if actual.size else np.nan,
            "actual_std": float(np.std(actual)) if actual.size else np.nan,
            "calibration_intercept": intercept, "calibration_slope": slope,
        }
    scores = [value["rolling_pearson_ic_mean"] for value in assets.values()]
    finite = [value for value in scores if np.isfinite(value)]
    sequence = pd.concat(sequence_parts, ignore_index=True) if sequence_parts else pd.DataFrame()
    return {
        "selection_metric": "min_asset_rolling_pearson_ic_mean",
        "selection_score": float(min(finite)) if len(finite) == len(assets) and assets else np.nan,
        "resampling_minutes": int(resampling_minutes),
        "roll_window": int(roll_window), "min_periods": int(min_periods),
        "assets": assets, "ic_sequence": sequence}


__all__ = ["evaluate_predictions"]
