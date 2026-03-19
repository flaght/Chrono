import json
import os
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    x1 = x[mask]
    y1 = y[mask]
    if float(x1.std(ddof=0)) == 0.0 or float(y1.std(ddof=0)) == 0.0:
        return 0.0
    v = x1.corr(y1, method=method)
    return float(v) if pd.notna(v) else 0.0


def _icir(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0
    std = float(s.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(s.mean() / std)


def _forward_sum(arr: np.ndarray, horizon: int) -> np.ndarray:
    """
    计算从当前步开始、长度为 horizon 的前向累计和。
    末尾不足 horizon 的位置返回 NaN。
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if horizon <= 0 or n < horizon:
        return out
    kernel = np.ones(horizon, dtype=np.float64)
    valid = np.convolve(arr, kernel, mode="valid")
    out[: len(valid)] = valid
    return out


def _daily_ic_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    基于包含 trade_time/action_eval/future_ret_h 的子样本，计算日度 IC 与 ICIR。
    """
    out = {
        "daily_pearson_ic": 0.0,
        "daily_spearman_rank_ic": 0.0,
        "daily_pearson_icir": 0.0,
        "daily_spearman_rank_icir": 0.0,
    }
    if "trade_time" not in df.columns or len(df) == 0:
        return out

    tmp_day = df[["trade_time", "action_eval", "future_ret_h"]].copy()
    tmp_day["trade_time"] = pd.to_datetime(tmp_day["trade_time"], errors="coerce")
    tmp_day = tmp_day.dropna(subset=["trade_time"])
    if len(tmp_day) == 0:
        return out

    tmp_day["trade_date"] = tmp_day["trade_time"].dt.date
    ic_list = []
    rank_ic_list = []
    for _, g in tmp_day.groupby("trade_date"):
        ic_list.append(_safe_corr(g["action_eval"], g["future_ret_h"], method="pearson"))
        rank_ic_list.append(_safe_corr(g["action_eval"], g["future_ret_h"], method="spearman"))
    ic_s = pd.Series(ic_list, dtype=float)
    rank_s = pd.Series(rank_ic_list, dtype=float)
    if len(ic_s) == 0:
        return out

    out["daily_pearson_ic"] = float(ic_s.mean())
    out["daily_spearman_rank_ic"] = float(rank_s.mean())
    out["daily_pearson_icir"] = _icir(ic_s)
    out["daily_spearman_rank_icir"] = _icir(rank_s)
    return out


def _subset_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算某个子样本（通常是阈值过滤后）的预测力指标。
    """
    if len(df) == 0:
        return {
            "samples": 0,
            "coverage": 0.0,
            "pearson_ic": 0.0,
            "spearman_rank_ic": 0.0,
            "sign_accuracy": 0.0,
            "mean_signed_ret_h": 0.0,
            "mean_action_x_ret_h": 0.0,
            "long_short_spread_q90_q10": 0.0,
            "daily_pearson_icir": 0.0,
            "daily_spearman_rank_icir": 0.0,
        }

    action = df["action_eval"].to_numpy(dtype=np.float64)
    label = df["future_ret_h"].to_numpy(dtype=np.float64)
    sign_acc = float((np.sign(action) == np.sign(label)).mean())
    mean_signed = float((np.sign(action) * label).mean())
    mean_axr = float((action * label).mean())

    q_hi = float(df["action_eval"].quantile(0.9))
    q_lo = float(df["action_eval"].quantile(0.1))
    high_mean = float(df.loc[df["action_eval"] >= q_hi, "future_ret_h"].mean())
    low_mean = float(df.loc[df["action_eval"] <= q_lo, "future_ret_h"].mean())
    spread = high_mean - low_mean

    daily = _daily_ic_stats(df)
    return {
        "samples": int(len(df)),
        "coverage": 0.0,  # 由调用方补充
        "pearson_ic": float(_safe_corr(df["action_eval"], df["future_ret_h"], method="pearson")),
        "spearman_rank_ic": float(_safe_corr(df["action_eval"], df["future_ret_h"], method="spearman")),
        "sign_accuracy": float(sign_acc),
        "mean_signed_ret_h": float(mean_signed),
        "mean_action_x_ret_h": float(mean_axr),
        "long_short_spread_q90_q10": float(spread),
        "daily_pearson_icir": float(daily["daily_pearson_icir"]),
        "daily_spearman_rank_icir": float(daily["daily_spearman_rank_icir"]),
    }


def _threshold_scan(eval_df: pd.DataFrame) -> list:
    """
    按 |action| 分位数做阈值扫描，便于直接选交易阈值。
    """
    if len(eval_df) == 0:
        return []

    quantiles = [0.70, 0.80, 0.90, 0.95]
    base_n = len(eval_df)
    out = []

    for q in quantiles:
        thr = float(eval_df["action_eval"].abs().quantile(q))
        sub = eval_df.loc[eval_df["action_eval"].abs() >= thr].copy()
        m = _subset_metrics(sub)
        m["coverage"] = float(len(sub) / base_n) if base_n > 0 else 0.0
        m["quantile"] = float(q)
        m["abs_action_threshold"] = float(thr)
        out.append(m)

    return out


def _build_trade_gate(
    spread: float,
    bucket_monotonicity: float,
    threshold_scan: list,
) -> Dict[str, Any]:
    """
    交易阈值落地门槛：
    1) 全样本 long-short spread > 0
    2) 分桶单调性 > 0
    满足后再从阈值扫描中选候选阈值。
    """
    gate_pass = bool((spread > 0.0) and (bucket_monotonicity > 0.0))
    out: Dict[str, Any] = {
        "gate_rule": "long_short_spread_q90_q10 > 0 and bucket_monotonicity_spearman > 0",
        "gate_passed": gate_pass,
        # 最终可落地条件：通过门槛 + 找到可用阈值
        "eligible_for_threshold_deployment": False,
        "recommended_threshold": None,
        "reason": "",
    }
    if not gate_pass:
        out["reason"] = (
            f"未通过门槛: spread={spread:.6g}, "
            f"bucket_monotonicity={bucket_monotonicity:.6g}"
        )
        return out

    candidates = [
        row for row in threshold_scan
        if (row.get("long_short_spread_q90_q10", 0.0) > 0.0)
        and (row.get("mean_action_x_ret_h", 0.0) > 0.0)
    ]
    if not candidates:
        out["reason"] = "全样本通过门槛，但阈值扫描子样本没有同时满足正向 spread 与正向 mean_action_x_ret_h 的候选。"
        return out

    best = sorted(
        candidates,
        key=lambda x: (
            float(x.get("mean_action_x_ret_h", 0.0)),
            float(x.get("spearman_rank_ic", 0.0)),
            float(x.get("coverage", 0.0)),
        ),
        reverse=True,
    )[0]
    out["recommended_threshold"] = {
        "quantile": float(best["quantile"]),
        "abs_action_threshold": float(best["abs_action_threshold"]),
        "coverage": float(best["coverage"]),
        "mean_action_x_ret_h": float(best["mean_action_x_ret_h"]),
        "spearman_rank_ic": float(best["spearman_rank_ic"]),
        "long_short_spread_q90_q10": float(best["long_short_spread_q90_q10"]),
    }
    out["eligible_for_threshold_deployment"] = True
    out["reason"] = "已通过全样本门槛，并在阈值扫描中选择 mean_action_x_ret_h 最优候选。"
    return out


def _build_action_series(df: pd.DataFrame) -> pd.Series:
    if "action_raw" in df.columns:
        return pd.to_numeric(df["action_raw"], errors="coerce").astype(float)
    if "er" in df.columns:
        return pd.to_numeric(df["er"], errors="coerce").astype(float)
    if "signal" in df.columns:
        return pd.to_numeric(df["signal"], errors="coerce").astype(float)
    if "action" in df.columns:
        # 兼容字符串格式，例如 "[0.123]"
        a = (
            df["action"]
            .astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
        )
        return pd.to_numeric(a, errors="coerce").astype(float)
    raise ValueError("results.csv 缺少 action_raw/er/signal/action 字段，无法评估预测力")


def _predictive_report(data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    if "current_ret" not in data.columns:
        raise ValueError("results.csv 缺少 current_ret 列，无法构造未来累计收益标签")

    df = data.copy()
    action = _build_action_series(df)
    if "future_ret_h" in df.columns:
        future_h = pd.to_numeric(df["future_ret_h"], errors="coerce").astype(float).to_numpy(dtype=np.float64)
        # 兜底：若历史文件未写入 future_ret_h，则按 current_ret 现算
        if int(np.isfinite(future_h).sum()) == 0:
            ret1 = pd.to_numeric(df["current_ret"], errors="coerce").astype(float)
            future_h = _forward_sum(ret1.to_numpy(dtype=np.float64), horizon=horizon)
    else:
        ret1 = pd.to_numeric(df["current_ret"], errors="coerce").astype(float)
        future_h = _forward_sum(ret1.to_numpy(dtype=np.float64), horizon=horizon)
    df["action_eval"] = action
    df["future_ret_h"] = future_h

    # 仅评估真正开仓样本（与训练目标一致）
    if "opened" in df.columns:
        opened_mask = df["opened"].astype(bool)
    else:
        opened_mask = pd.Series(np.ones(len(df), dtype=bool), index=df.index)

    mask = (
        opened_mask
        & df["action_eval"].notna()
        & df["future_ret_h"].notna()
        & np.isfinite(df["action_eval"])
        & np.isfinite(df["future_ret_h"])
    )
    eval_df = df.loc[mask, ["action_eval", "future_ret_h"]].copy()
    if eval_df.empty:
        return {
            "samples_total": int(len(df)),
            "samples_opened": int(opened_mask.sum()),
            "samples_effective": 0,
            "horizon": int(horizon),
            "pearson_ic": 0.0,
            "spearman_rank_ic": 0.0,
            "sign_accuracy": 0.0,
            "long_short_spread_q90_q10": 0.0,
            "bucket_means": [],
            "threshold_scan_abs_action_quantiles": [],
        }

    pearson_ic = _safe_corr(eval_df["action_eval"], eval_df["future_ret_h"], method="pearson")
    rank_ic = _safe_corr(eval_df["action_eval"], eval_df["future_ret_h"], method="spearman")

    sign_acc = float(
        (
            np.sign(eval_df["action_eval"].to_numpy(dtype=np.float64))
            == np.sign(eval_df["future_ret_h"].to_numpy(dtype=np.float64))
        ).mean()
    )

    # 分桶评估单调性 + 多空差
    n_bins = 10
    tmp = eval_df.copy()
    try:
        tmp["bucket"] = pd.qcut(tmp["action_eval"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        tmp["bucket"] = 0
    bucket_stats = (
        tmp.groupby("bucket", dropna=True)["future_ret_h"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("bucket")
    )
    bucket_means = [
        {
            "bucket": int(r["bucket"]),
            "mean_future_ret_h": float(r["mean"]),
            "count": int(r["count"]),
        }
        for _, r in bucket_stats.iterrows()
    ]

    q_hi = float(tmp["action_eval"].quantile(0.9))
    q_lo = float(tmp["action_eval"].quantile(0.1))
    high_mean = float(tmp.loc[tmp["action_eval"] >= q_hi, "future_ret_h"].mean())
    low_mean = float(tmp.loc[tmp["action_eval"] <= q_lo, "future_ret_h"].mean())
    spread = high_mean - low_mean

    # |action| 高置信子集（前20%）命中率/均值
    q_abs = float(eval_df["action_eval"].abs().quantile(0.8))
    top_abs = eval_df.loc[eval_df["action_eval"].abs() >= q_abs]
    if len(top_abs) > 0:
        top_abs_sign_acc = float(
            (
                np.sign(top_abs["action_eval"].to_numpy(dtype=np.float64))
                == np.sign(top_abs["future_ret_h"].to_numpy(dtype=np.float64))
            ).mean()
        )
        top_abs_mean = float(top_abs["future_ret_h"].mean())
    else:
        top_abs_sign_acc = 0.0
        top_abs_mean = 0.0

    # 分桶单调性评分：桶号 vs 桶均值的 Spearman
    bucket_monotonicity = 0.0
    if len(bucket_stats) >= 3:
        b = pd.to_numeric(bucket_stats["bucket"], errors="coerce")
        m = pd.to_numeric(bucket_stats["mean"], errors="coerce")
        bucket_monotonicity = _safe_corr(b, m, method="spearman")

    # 日度 IC / ICIR
    eval_with_time = eval_df.copy()
    if "trade_time" in df.columns:
        eval_with_time["trade_time"] = df.loc[mask, "trade_time"].values
    daily = _daily_ic_stats(eval_with_time)

    # 阈值扫描（|action| 分位数）
    threshold_scan = _threshold_scan(eval_with_time)

    trade_gate = _build_trade_gate(
        spread=float(spread),
        bucket_monotonicity=float(bucket_monotonicity),
        threshold_scan=threshold_scan,
    )

    return {
        "samples_total": int(len(df)),
        "samples_opened": int(opened_mask.sum()),
        "samples_effective": int(len(eval_df)),
        "horizon": int(horizon),
        "pearson_ic": float(pearson_ic),
        "spearman_rank_ic": float(rank_ic),
        "daily_pearson_ic": float(daily["daily_pearson_ic"]),
        "daily_spearman_rank_ic": float(daily["daily_spearman_rank_ic"]),
        "daily_pearson_icir": float(daily["daily_pearson_icir"]),
        "daily_spearman_rank_icir": float(daily["daily_spearman_rank_icir"]),
        "sign_accuracy": float(sign_acc),
        "top20_abs_action_sign_accuracy": float(top_abs_sign_acc),
        "top20_abs_action_mean_future_ret_h": float(top_abs_mean),
        "long_short_spread_q90_q10": float(spread),
        "bucket_monotonicity_spearman": float(bucket_monotonicity),
        "bucket_means": bucket_means,
        "threshold_scan_abs_action_quantiles": threshold_scan,
        "threshold_deployment_gate": trade_gate,
    }


def analyze_run(
    run_dir: str,
    output_path: str,
    annual_trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    RL011 预测力分析：聚焦 action -> 未来收益的可预测性。
    """
    del annual_trading_days, risk_free_rate

    results_path = os.path.join(run_dir, "metrics", "results.csv")
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.csv 不存在: {results_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json 不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    horizon = int(cfg.get("env_config", {}).get("holding_period", 15) or 15)

    data = pd.read_csv(results_path)
    predictive = _predictive_report(data, horizon=horizon)

    report = {
        "run_dir": run_dir,
        "created_at": datetime.now().isoformat(),
        "analysis_type": "rl011_action_predictive_power",
        "predictive_power": predictive,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
