import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from lib.logger import logger


def _read_monitor_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, comment="#")


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.lower()
    return s.isin(["true", "1", "yes", "y"])


def _summarize_eval_npz(eval_npz_path: str) -> Dict[str, Any]:
    if not os.path.exists(eval_npz_path):
        return {"exists": False}

    npz = np.load(eval_npz_path, allow_pickle=True)
    summary: Dict[str, Any] = {"exists": True, "keys": list(npz.files)}
    if "timesteps" in npz.files and "results" in npz.files:
        ts = npz["timesteps"]
        rs = npz["results"]
        mean = rs.mean(axis=1)
        std = rs.std(axis=1)
        best_i = int(mean.argmax())
        worst_i = int(mean.argmin())
        summary.update(
            {
                "n_eval": int(len(ts)),
                "first_ts": int(ts[0]),
                "last_ts": int(ts[-1]),
                "best_mean": float(mean[best_i]),
                "best_ts": int(ts[best_i]),
                "best_std": float(std[best_i]),
                "worst_mean": float(mean[worst_i]),
                "worst_ts": int(ts[worst_i]),
                "worst_std": float(std[worst_i]),
                "last_mean": float(mean[-1]),
                "last_std": float(std[-1]),
            }
        )
    return summary


def _summarize_monitor(path: str) -> Dict[str, Any]:
    df = _read_monitor_csv(path)
    if df is None or len(df) == 0:
        return {"exists": False}
    r = df["r"].astype(float)
    out: Dict[str, Any] = {
        "exists": True,
        "rows": int(len(df)),
        "min": float(r.min()),
        "max": float(r.max()),
        "mean": float(r.mean()),
        "last": float(r.iloc[-1]),
    }
    if len(df) >= 10:
        out["rolling10_last"] = float(r.rolling(10).mean().iloc[-1])
    out["unique_rewards"] = int(r.nunique())
    return out


def _summarize_results(df: pd.DataFrame) -> Dict[str, Any]:
    vc = df["direction"].value_counts().to_dict()
    n = len(df)
    n_long = int(vc.get(1, 0))
    n_short = int(vc.get(-1, 0))
    n_flat = int(vc.get(0, 0))

    opened = int(_to_bool_series(df["opened"]).sum()) if "opened" in df.columns else 0
    active_max = int(df["active_signals"].max()) if "active_signals" in df.columns else 0
    net_min = int(df["net_position"].min()) if "net_position" in df.columns else 0
    net_max = int(df["net_position"].max()) if "net_position" in df.columns else 0

    conf = df["confidence"].astype(float) if "confidence" in df.columns else pd.Series(np.zeros(n))
    nz = conf > 0
    conf_nz_mean = float(conf[nz].mean()) if nz.any() else 0.0

    reward = df["reward"].astype(float) if "reward" in df.columns else pd.Series(np.zeros(n))
    reward_scaled = df["reward_scaled"].astype(float) if "reward_scaled" in df.columns else reward * 10000.0

    nonzero_dir = df.loc[df["direction"] != 0, "direction"].astype(int)
    flips = int((nonzero_dir.diff().abs() == 2).sum()) if len(nonzero_dir) > 1 else 0
    flip_ratio = float(flips / max(1, len(nonzero_dir) - 1))

    equity = reward_scaled.cumsum()
    drawdown = equity - equity.cummax()

    return {
        "rows": int(n),
        "first_time": str(df["trade_time"].iloc[0]),
        "last_time": str(df["trade_time"].iloc[-1]),
        "long_count": n_long,
        "short_count": n_short,
        "flat_count": n_flat,
        "long_ratio": float(n_long / n),
        "short_ratio": float(n_short / n),
        "flat_ratio": float(n_flat / n),
        "opened": opened,
        "active_max": active_max,
        "net_pos_min": net_min,
        "net_pos_max": net_max,
        "confidence_mean": float(conf.mean()),
        "confidence_nonzero_mean": conf_nz_mean,
        "reward_sum": float(reward.sum()),
        "reward_mean": float(reward.mean()),
        "reward_scaled_sum": float(reward_scaled.sum()),
        "nonzero_signals": int(len(nonzero_dir)),
        "flips": flips,
        "flip_ratio": flip_ratio,
        "equity_scaled_end": float(equity.iloc[-1]),
        "max_drawdown_scaled": float(drawdown.min()),
    }


def _single_asset_performance(
    df: pd.DataFrame,
    config: Dict[str, Any],
    annual_trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    data = df.copy()
    data["trade_time"] = pd.to_datetime(data["trade_time"])
    data = data.sort_values("trade_time").reset_index(drop=True)
    data["trade_date"] = data["trade_time"].dt.date

    if "reward" in data.columns:
        step_ret = data["reward"].astype(float)
    elif "reward_raw" in data.columns:
        step_ret = data["reward_raw"].astype(float)
    else:
        step_ret = (
            data.get("direction", 0).astype(float)
            * data.get("confidence", 0.0).astype(float)
            * data.get("current_ret", 0.0).astype(float)
        )

    daily_returns = step_ret.groupby(data["trade_date"]).apply(lambda x: float((1.0 + x).prod() - 1.0))
    test_days = int(len(daily_returns))
    cumulative_return = float((1.0 + daily_returns).prod() - 1.0) if test_days > 0 else 0.0

    if test_days > 0:
        annualized_return = float((1.0 + cumulative_return) ** (annual_trading_days / test_days) - 1.0)
        annualized_vol = float(daily_returns.std(ddof=0) * np.sqrt(annual_trading_days))
    else:
        annualized_return = 0.0
        annualized_vol = 0.0

    rf_daily = risk_free_rate / annual_trading_days
    excess_daily = daily_returns - rf_daily
    if len(excess_daily) > 1 and float(excess_daily.std(ddof=0)) > 0.0:
        sharpe = float(excess_daily.mean() / excess_daily.std(ddof=0) * np.sqrt(annual_trading_days))
    else:
        sharpe = 0.0

    eq = (1.0 + daily_returns).cumprod()
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    daily_win_rate = float((daily_returns > 0).mean()) if test_days > 0 else 0.0
    avg_win = float(daily_returns[daily_returns > 0].mean()) if (daily_returns > 0).any() else 0.0
    avg_loss = float(daily_returns[daily_returns < 0].mean()) if (daily_returns < 0).any() else 0.0
    daily_pl_ratio = float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0

    if "turnover" in data.columns:
        turnover_step = data["turnover"].astype(float)
    else:
        # 单资产标准换手口径：0.5 * |w_t - w_{t-1}|, w in [-1, 1]
        max_pos = float(config.get("env_config", {}).get("max_allowed_position", 1.0) or 1.0)
        net_pos = data.get("net_position", pd.Series(np.zeros(len(data)))).astype(float)
        weight = net_pos / max(max_pos, 1.0)
        turnover_step = 0.5 * weight.diff().abs().fillna(weight.abs())
    daily_turnover = turnover_step.groupby(data["trade_date"]).sum()
    avg_daily_turnover = float(daily_turnover.mean()) if len(daily_turnover) > 0 else 0.0
    total_turnover = float(daily_turnover.sum()) if len(daily_turnover) > 0 else 0.0
    bars_per_day = data.groupby("trade_date").size()
    avg_daily_turnover_per_bar = float(
        (daily_turnover / bars_per_day).replace([np.inf, -np.inf], np.nan).dropna().mean()
    ) if len(daily_turnover) > 0 else 0.0

    sig_cfg = config.get("signal_config", {})
    base_cost = float(sig_cfg.get("base_cost", 0.0) or 0.0)
    if "trade_cost" in data.columns:
        step_cost = data["trade_cost"].astype(float)
    else:
        opened = _to_bool_series(data.get("opened", pd.Series([False] * len(data)))).astype(float)
        expired = data.get("expired_count", pd.Series(np.zeros(len(data)))).astype(float)
        step_cost = (opened + expired) * base_cost
    daily_cost = step_cost.groupby(data["trade_date"]).sum()
    avg_daily_cost = float(daily_cost.mean()) if len(daily_cost) > 0 else 0.0
    total_cost = float(daily_cost.sum()) if len(daily_cost) > 0 else 0.0

    if "n_holdings" in data.columns:
        holding_count = data["n_holdings"].astype(float)
    else:
        holding_count = data.get("active_signals", pd.Series(np.zeros(len(data)))).astype(float)
    avg_daily_holding_count = float(holding_count.groupby(data["trade_date"]).mean().mean())

    if "hhi" in data.columns:
        hhi = data["hhi"].astype(float)
    else:
        net_abs = data.get("net_position", pd.Series(np.zeros(len(data)))).astype(float).abs()
        hhi = (net_abs > 0).astype(float)
    avg_daily_hhi = float(hhi.groupby(data["trade_date"]).mean().mean())

    return {
        "test_days": test_days,
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "daily_win_rate": daily_win_rate,
        "daily_profit_loss_ratio": daily_pl_ratio,
        "avg_daily_turnover": avg_daily_turnover,
        "total_turnover": total_turnover,
        "avg_daily_turnover_per_bar": avg_daily_turnover_per_bar,
        "avg_daily_trade_cost": avg_daily_cost,
        "total_trade_cost": total_cost,
        "avg_daily_holding_count": avg_daily_holding_count,
        "avg_daily_hhi": avg_daily_hhi,
    }


def _build_step_pnl_columns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    构建单步收益分解：
    net_step_return = reward
    gross_step_return = net + estimated_cost + estimated_flip_penalty
    """
    data = df.copy()
    if "trade_time" not in data.columns:
        raise ValueError("results.csv 缺少 trade_time 列，无法进行按日绩效分析")
    data["trade_time"] = pd.to_datetime(data["trade_time"])
    data = data.sort_values("trade_time").reset_index(drop=True)
    data["trade_date"] = data["trade_time"].dt.date
    n = len(data)
    sig_cfg = config.get("signal_config", {})
    base_cost = float(sig_cfg.get("base_cost", 0.0) or 0.0)
    cost_mode = str(sig_cfg.get("cost_mode", "fixed"))

    if "reward" in data.columns:
        net_step = data["reward"].astype(float)
    elif "reward_raw" in data.columns:
        net_step = data["reward_raw"].astype(float)
    else:
        net_step = (
            data.get("direction", 0).astype(float)
            * data.get("confidence", 0.0).astype(float)
            * data.get("current_ret", 0.0).astype(float)
        )

    opened = _to_bool_series(data.get("opened", pd.Series([False] * n))).astype(float)
    expired = data.get("expired_count", pd.Series(np.zeros(n))).astype(float)

    if "trade_cost" in data.columns:
        est_cost_step = data["trade_cost"].astype(float)
    else:
        if cost_mode == "fixed":
            est_cost_step = (opened + expired) * base_cost
        else:
            # proportional 模式在当前结果中无法精确重建平仓成本，用开仓信号近似
            signal_abs = data.get("signal", pd.Series(np.zeros(n))).astype(float).abs()
            est_cost_step = opened * base_cost * signal_abs

    # 方向翻转惩罚：环境硬编码 0.0005
    dir_s = data.get("direction", pd.Series(np.zeros(n))).astype(int)
    prev_nonzero = dir_s.replace(0, np.nan).ffill().shift(1).fillna(0).astype(int)
    flip_mask = (dir_s != 0) & (prev_nonzero != 0) & (dir_s != prev_nonzero)
    est_flip_penalty_step = flip_mask.astype(float) * 0.0005

    data["net_step_return"] = net_step
    data["estimated_cost_step"] = est_cost_step
    data["estimated_flip_penalty_step"] = est_flip_penalty_step
    data["gross_step_return"] = net_step + est_cost_step + est_flip_penalty_step
    return data


def _daily_return_from_step(step_returns: pd.Series, trade_dates: pd.Series) -> pd.Series:
    return step_returns.groupby(trade_dates).apply(lambda x: float((1.0 + x).prod() - 1.0))


def _return_metrics(daily_returns: pd.Series, annual_trading_days: int, risk_free_rate: float) -> Dict[str, Any]:
    n = int(len(daily_returns))
    if n == 0:
        return {
            "days": 0,
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "daily_win_rate": 0.0,
            "daily_profit_loss_ratio": 0.0,
        }

    cum_ret = float((1.0 + daily_returns).prod() - 1.0)
    ann_ret = float((1.0 + cum_ret) ** (annual_trading_days / n) - 1.0)
    ann_vol = float(daily_returns.std(ddof=0) * np.sqrt(annual_trading_days))

    rf_daily = risk_free_rate / annual_trading_days
    excess = daily_returns - rf_daily
    sharpe = float(excess.mean() / excess.std(ddof=0) * np.sqrt(annual_trading_days)) if len(excess) > 1 and float(excess.std(ddof=0)) > 0 else 0.0

    eq = (1.0 + daily_returns).cumprod()
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min()) if len(dd) > 0 else 0.0

    win_rate = float((daily_returns > 0).mean())
    avg_win = float(daily_returns[daily_returns > 0].mean()) if (daily_returns > 0).any() else 0.0
    avg_loss = float(daily_returns[daily_returns < 0].mean()) if (daily_returns < 0).any() else 0.0
    pl_ratio = float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0

    return {
        "days": n,
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "daily_win_rate": win_rate,
        "daily_profit_loss_ratio": pl_ratio,
    }


def _benchmark_summary(data: pd.DataFrame, annual_trading_days: int, risk_free_rate: float) -> Dict[str, Any]:
    daily_net = _daily_return_from_step(data["net_step_return"], data["trade_date"])
    daily_gross = _daily_return_from_step(data["gross_step_return"], data["trade_date"])
    daily_cost = data["estimated_cost_step"].groupby(data["trade_date"]).sum()

    buy_hold_daily = _daily_return_from_step(data.get("current_ret", pd.Series(np.zeros(len(data)))).astype(float), data["trade_date"])
    flat_daily = pd.Series(np.zeros(len(daily_net)), index=daily_net.index)

    # 对齐
    idx = daily_net.index
    buy_hold_daily = buy_hold_daily.reindex(idx).fillna(0.0)
    flat_daily = flat_daily.reindex(idx).fillna(0.0)
    daily_gross = daily_gross.reindex(idx).fillna(0.0)
    daily_cost = daily_cost.reindex(idx).fillna(0.0)

    strategy_metrics = _return_metrics(daily_net, annual_trading_days, risk_free_rate)
    gross_metrics = _return_metrics(daily_gross, annual_trading_days, risk_free_rate)
    buy_hold_metrics = _return_metrics(buy_hold_daily, annual_trading_days, risk_free_rate)
    flat_metrics = _return_metrics(flat_daily, annual_trading_days, risk_free_rate)

    excess_vs_bh = daily_net - buy_hold_daily
    excess_vs_flat = daily_net - flat_daily
    excess_bh_metrics = _return_metrics(excess_vs_bh, annual_trading_days, risk_free_rate)
    excess_flat_metrics = _return_metrics(excess_vs_flat, annual_trading_days, risk_free_rate)

    return {
        "strategy_net": strategy_metrics,
        "strategy_gross": gross_metrics,
        "daily_cost_mean": float(daily_cost.mean()) if len(daily_cost) > 0 else 0.0,
        "daily_cost_sum": float(daily_cost.sum()) if len(daily_cost) > 0 else 0.0,
        "buy_and_hold": buy_hold_metrics,
        "always_flat": flat_metrics,
        "excess_vs_buy_and_hold": excess_bh_metrics,
        "excess_vs_flat": excess_flat_metrics,
    }


def _stability_by_period(data: pd.DataFrame) -> Dict[str, Any]:
    daily_net = _daily_return_from_step(data["net_step_return"], data["trade_date"])
    if len(daily_net) == 0:
        return {"monthly": [], "quarterly": []}

    s = daily_net.copy()
    s.index = pd.to_datetime(s.index)

    monthly = []
    for p, v in s.groupby(s.index.to_period("M")):
        cum = float((1.0 + v).prod() - 1.0)
        win = float((v > 0).mean()) if len(v) > 0 else 0.0
        vol = float(v.std(ddof=0))
        monthly.append(
            {
                "period": str(p),
                "days": int(len(v)),
                "return": cum,
                "win_rate": win,
                "volatility": vol,
            }
        )

    quarterly = []
    for p, v in s.groupby(s.index.to_period("Q")):
        cum = float((1.0 + v).prod() - 1.0)
        win = float((v > 0).mean()) if len(v) > 0 else 0.0
        vol = float(v.std(ddof=0))
        quarterly.append(
            {
                "period": str(p),
                "days": int(len(v)),
                "return": cum,
                "win_rate": win,
                "volatility": vol,
            }
        )

    return {"monthly": monthly, "quarterly": quarterly}


def _trade_level_stats(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    交易级统计：基于 fixed holding_period 的近似逐笔收益重建。
    """
    holding_period = int(config.get("env_config", {}).get("holding_period", 1) or 1)
    base_cost = float(config.get("signal_config", {}).get("base_cost", 0.0) or 0.0)
    opened_mask = _to_bool_series(data.get("opened", pd.Series([False] * len(data))))

    entries = data.loc[opened_mask].copy()
    if len(entries) == 0:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "profit_loss_ratio": 0.0,
            "avg_holding_steps": float(holding_period),
            "flip_trade_ratio": 0.0,
        }

    cur_ret = data.get("current_ret", pd.Series(np.zeros(len(data)))).astype(float).values
    trade_returns = []
    entry_dirs = entries.get("direction", pd.Series(np.zeros(len(entries)))).astype(float).values
    entry_conf = entries.get("confidence", pd.Series(np.ones(len(entries)))).astype(float).values
    entry_idx = entries.index.values

    for i, d, c in zip(entry_idx, entry_dirs, entry_conf):
        j = min(i + holding_period, len(data) - 1)
        gross = float(d * c * np.sum(cur_ret[i:j + 1]))
        net = gross - 2.0 * base_cost
        trade_returns.append(net)

    trade_s = pd.Series(trade_returns, dtype=float)
    win_rate = float((trade_s > 0).mean()) if len(trade_s) > 0 else 0.0
    avg_ret = float(trade_s.mean()) if len(trade_s) > 0 else 0.0
    avg_win = float(trade_s[trade_s > 0].mean()) if (trade_s > 0).any() else 0.0
    avg_loss = float(trade_s[trade_s < 0].mean()) if (trade_s < 0).any() else 0.0
    pl_ratio = float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0

    d_nonzero = data.loc[data["direction"] != 0, "direction"].astype(int)
    flips = int((d_nonzero.diff().abs() == 2).sum()) if len(d_nonzero) > 1 else 0
    flip_ratio = float(flips / max(1, len(d_nonzero) - 1))

    return {
        "trade_count": int(len(trade_s)),
        "win_rate": win_rate,
        "avg_trade_return": avg_ret,
        "profit_loss_ratio": pl_ratio,
        "avg_holding_steps": float(holding_period),
        "flip_trade_ratio": flip_ratio,
    }


def _risk_exposure_stats(data: pd.DataFrame, config: Dict[str, Any], annual_trading_days: int) -> Dict[str, Any]:
    max_pos = float(config.get("env_config", {}).get("max_allowed_position", 1.0) or 1.0)
    net_pos = data.get("net_position", pd.Series(np.zeros(len(data)))).astype(float)
    exposure = net_pos / max(max_pos, 1.0)

    daily_net = _daily_return_from_step(data["net_step_return"], data["trade_date"])
    if len(daily_net) > 0:
        sorted_r = np.sort(daily_net.values)
        q_idx = max(int(np.floor(0.05 * len(sorted_r))) - 1, 0)
        var95 = float(sorted_r[q_idx])
        cvar95 = float(sorted_r[sorted_r <= var95].mean()) if np.any(sorted_r <= var95) else var95
    else:
        var95 = 0.0
        cvar95 = 0.0

    # 最大连续亏损天数
    max_losing_streak = 0
    cur_streak = 0
    for r in daily_net.values:
        if r < 0:
            cur_streak += 1
            max_losing_streak = max(max_losing_streak, cur_streak)
        else:
            cur_streak = 0

    return {
        "avg_net_exposure": float(exposure.mean()),
        "avg_abs_exposure": float(exposure.abs().mean()),
        "max_abs_exposure": float(exposure.abs().max()) if len(exposure) > 0 else 0.0,
        "var_95_daily": var95,
        "cvar_95_daily": cvar95,
        "max_consecutive_losing_days": int(max_losing_streak),
    }


def analyze_run(
    run_dir: str,
    output_path: Optional[str] = None,
    annual_trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    解析 rl001 一次训练/预测结果目录，返回综合分析。

    Args:
        run_dir: 结果目录，内部包含 config.json、logs/、metrics/results.csv
        output_path: 可选，分析结果 JSON 保存路径
        annual_trading_days: 年化交易日数量
        risk_free_rate: 年化无风险利率
    """
    run_dir = os.path.abspath(run_dir)
    config_path = os.path.join(run_dir, "config.json")
    eval_npz_path = os.path.join(run_dir, "logs", "eval", "evaluations.npz")
    train_monitor_path = os.path.join(run_dir, "logs", "train_monitor.csv")
    val_monitor_path = os.path.join(run_dir, "logs", "val_monitor.csv")
    results_path = os.path.join(run_dir, "metrics", "results.csv")
    training_metrics_path = os.path.join(run_dir, "logs", "training_metrics.json")

    config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    analysis: Dict[str, Any] = {
        "run_dir": run_dir,
        "config_exists": os.path.exists(config_path),
        "eval": _summarize_eval_npz(eval_npz_path),
        "train_monitor": _summarize_monitor(train_monitor_path),
        "val_monitor": _summarize_monitor(val_monitor_path),
        "results_exists": os.path.exists(results_path),
        "training_metrics_exists": os.path.exists(training_metrics_path),
    }

    # 质量检查：eval episode 数过小时提示
    eval_checks: Dict[str, Any] = {}
    if analysis["eval"].get("exists"):
        eval_ep_count = None
        try:
            npz = np.load(eval_npz_path, allow_pickle=True)
            if "results" in npz.files:
                eval_ep_count = int(npz["results"].shape[1]) if npz["results"].ndim >= 2 else 1
        except Exception:
            eval_ep_count = None
        eval_checks = {
            "eval_episodes_per_checkpoint": eval_ep_count,
            "eval_episodes_recommended_min": 5,
            "is_eval_episode_count_sufficient": bool(eval_ep_count is not None and eval_ep_count >= 5),
        }
    analysis["quality_checks"] = eval_checks

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        if len(df) > 0:
            analysis["results"] = _summarize_results(df)

            step_data = _build_step_pnl_columns(df, config)
            analysis["single_asset_performance"] = _single_asset_performance(
                df=df,
                config=config,
                annual_trading_days=annual_trading_days,
                risk_free_rate=risk_free_rate,
            )
            analysis["pnl_decomposition_and_benchmarks"] = _benchmark_summary(
                data=step_data,
                annual_trading_days=annual_trading_days,
                risk_free_rate=risk_free_rate,
            )
            analysis["stability"] = _stability_by_period(step_data)
            analysis["trade_level"] = _trade_level_stats(step_data, config)
            analysis["risk_metrics"] = _risk_exposure_stats(
                data=step_data,
                config=config,
                annual_trading_days=annual_trading_days,
            )
        else:
            analysis["results"] = {"rows": 0}
            analysis["single_asset_performance"] = {}
            analysis["pnl_decomposition_and_benchmarks"] = {}
            analysis["stability"] = {"monthly": [], "quarterly": []}
            analysis["trade_level"] = {}
            analysis["risk_metrics"] = {}
    else:
        analysis["results"] = {"exists": False}
        analysis["single_asset_performance"] = {}
        analysis["pnl_decomposition_and_benchmarks"] = {}
        analysis["stability"] = {"monthly": [], "quarterly": []}
        analysis["trade_level"] = {}
        analysis["risk_metrics"] = {}

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        logger.info(f"分析结果已保存到: {output_path}")

    return analysis


def batch_analyze_runs(
    root_dir: str,
    output_csv: Optional[str] = None,
    annual_trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    遍历指定根目录下的所有子目录，对每一组训练/预测结果计算绩效值，并汇总输出为 DataFrame。
    
    Args:
        root_dir: 结果根目录（例如 temp/model/200037/15/rl/result）
        output_csv: 保存汇总 CSV 的路径（可选）
        annual_trading_days: 年化交易日数量
        risk_free_rate: 年化无风险利率
    
    Returns:
        pd.DataFrame: 包含所有测试跑批核心绩效指标和参数的汇总表，按 cumulative_return 降序排列。
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        logger.error(f"指定的根目录不存在或不是目录: {root_dir}")
        return pd.DataFrame()

    results_list = []
    
    # 遍历 root_dir 下的直接子目录
    for run_name in os.listdir(root_dir):
        run_dir_path = os.path.join(root_dir, run_name)
        if not os.path.isdir(run_dir_path):
            continue
            
        # 必须存在 metrics/results.csv 才认为是一次有效的 run
        results_path = os.path.join(run_dir_path, "metrics", "results.csv")
        if not os.path.exists(results_path):
            continue
            
        try:
            analysis = analyze_run(
                run_dir=run_dir_path,
                output_path=None,  # 批处理时不单独存 JSON
                annual_trading_days=annual_trading_days,
                risk_free_rate=risk_free_rate,
            )
            
            # 提取核心绩效指标
            flat_res = {
                "Run_ID": run_name,
            }
            
            # 从 single_asset_performance 提取
            perf = analysis.get("single_asset_performance", {})
            flat_res["Total_Return"] = perf.get("cumulative_return", 0.0)
            flat_res["Max_Drawdown"] = perf.get("max_drawdown", 0.0)
            flat_res["Sharpe_Ratio"] = perf.get("sharpe_ratio", 0.0)
            flat_res["Win_Rate"] = perf.get("daily_win_rate", 0.0)
            flat_res["PnL_Ratio"] = perf.get("daily_profit_loss_ratio", 0.0)
            
            # 从 results 提取交易统计
            base_res = analysis.get("results", {})
            flat_res["Total_Trades"] = base_res.get("nonzero_signals", 0)
            
            tl_res = analysis.get("trade_level", {})
            if "trade_count" in tl_res:
                flat_res["Total_Trades"] = tl_res["trade_count"]
                
            flat_res["Long_Count"] = base_res.get("long_count", 0)
            flat_res["Short_Count"] = base_res.get("short_count", 0)
            flat_res["Total_Cost"] = perf.get("total_trade_cost", 0.0)
            
            # 提取重要参数 (从 config.json 中)
            config_path = os.path.join(run_dir_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    sig_cfg = cfg.get("signal_config", {})
                    flat_res["Temp"] = sig_cfg.get("temperature", None)
                    flat_res["Thresh"] = sig_cfg.get("threshold", None)
                    flat_res["Thresh_Min"] = sig_cfg.get("threshold_min", None)
                    flat_res["Mapping"] = sig_cfg.get("score_mapping", None)
            
            results_list.append(flat_res)
            
        except Exception as e:
            logger.error(f"处理 {run_dir_path} 时发生错误: {e}")
            
    if not results_list:
        logger.warning(f"在 {root_dir} 下没有找到有效的测试结果。")
        return pd.DataFrame()
        
    df_summary = pd.DataFrame(results_list)
    # 按总收益率降序排序
    if "Total_Return" in df_summary.columns:
        df_summary = df_summary.sort_values(by="Total_Return", ascending=False).reset_index(drop=True)
        
    logger.info(f"批量分析完成，成功处理 {len(df_summary)} 个测试结果。")
    
    if output_csv:
        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
        logger.info(f"汇总结果已保存到: {output_csv}")
        
    return df_summary
