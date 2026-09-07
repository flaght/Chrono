import pdb
import pandas as pd
import numpy as np
from lib.attr001.ftd002 import *
from lib.rl012.analysis import profitability, quantile, pred_metrics


def safe_base(value, floor=1e-8):
    return max(abs(value), floor)


def build_horizon_returns(df,
                          ret_col="nxt1_ret",
                          return_name='future_ret_h',
                          holding_period=5):
    df = df.copy().sort_values(["trade_time", "code"]).reset_index(drop=True)

    def _calc_one(group):
        ret = pd.to_numeric(group[ret_col],
                            errors="coerce").astype(float).to_numpy()
        n = len(ret)
        out = np.full(n, np.nan, dtype=np.float64)

        if holding_period <= 0 or n < holding_period:
            group[return_name] = out
            return group

        # 和 RL 代码一致：NaN 先按 0 处理，再做 rolling sum
        ret2 = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        kernel = np.ones(holding_period, dtype=np.float64)
        valid = np.convolve(ret2, kernel, mode="valid")
        out[:len(valid)] = valid

        # 尾部不足 holding_period 的位置保留 NaN
        out[n - holding_period + 1:] = np.nan

        group[return_name] = out
        return group

    return df.groupby("code", group_keys=False).apply(_calc_one)


def judge_inspect_status(compare_row):
    ann_ret_rel_gap = compare_row["ann_ret_rel_gap"]
    ann_vol_rel_gap = compare_row["ann_vol_rel_gap"]
    sharpe_rel_gap = compare_row["sharpe_rel_gap"]
    win_rate_gap = abs(compare_row["win_rate_gap"])
    total_spread_gap = abs(compare_row["total_spread_gap"])
    turnover_gap = abs(compare_row["turnover_gap"])

    if (ann_ret_rel_gap <= 0.05 and ann_vol_rel_gap <= 0.05
            and sharpe_rel_gap <= 0.05 and win_rate_gap <= 0.01
            and total_spread_gap <= 0.002 and turnover_gap <= 0.02):
        return "PASS"

    if (ann_ret_rel_gap <= 0.10 and ann_vol_rel_gap <= 0.10
            and sharpe_rel_gap <= 0.10 and win_rate_gap <= 0.03
            and total_spread_gap <= 0.01 and turnover_gap <= 0.05):
        return "WARN"

    if (ann_ret_rel_gap <= 0.20 and ann_vol_rel_gap <= 0.20
            and sharpe_rel_gap <= 0.20 and win_rate_gap <= 0.05
            and total_spread_gap <= 0.02 and turnover_gap <= 0.10):
        return "FAIL_EDGE"

    return "FAIL"


def compare_summary(research_metrics, trader_metrics, factor_name):
    r_profit = research_metrics["summary"]["profit_results"]
    t_profit = trader_metrics["summary"]["profit_results"]
    r_spread = research_metrics["summary"]["spread_results"]
    t_spread = trader_metrics["summary"]["spread_results"]

    row = {
        "name": factor_name,
        "ann_ret_research": r_profit.get("ann_ret"),
        "ann_ret_trader": t_profit.get("ann_ret"),
        "ann_vol_research": r_profit.get("ann_vol"),
        "ann_vol_trader": t_profit.get("ann_vol"),
        "sharpe_research": r_profit.get("sharpe"),
        "sharpe_trader": t_profit.get("sharpe"),
        "calmar_research": r_profit.get("calmar"),
        "calmar_trader": t_profit.get("calmar"),
        "win_rate_research": r_profit.get("win_rate"),
        "win_rate_trader": t_profit.get("win_rate"),
        "profit_ratio_research": r_profit.get("profit_ratio"),
        "profit_ratio_trader": t_profit.get("profit_ratio"),
        "maxdd_research": r_profit.get("maxdd"),
        "maxdd_trader": t_profit.get("maxdd"),
        "turnover_research": r_profit.get("turnover"),
        "turnover_trader": t_profit.get("turnover"),
        "total_spread_research": r_spread.get("total_spread"),
        "total_spread_trader": t_spread.get("total_spread"),
        "spread_mean_research": r_spread.get("spread_mean"),
        "spread_mean_trader": t_spread.get("spread_mean"),
    }

    for metric in [
            "ann_ret", "ann_vol", "sharpe", "calmar", "win_rate",
            "profit_ratio", "maxdd", "turnover", "total_spread", "spread_mean"
    ]:
        row[f"{metric}_gap"] = row[f"{metric}_trader"] - row[
            f"{metric}_research"]

    row["ann_ret_rel_gap"] = abs(row["ann_ret_gap"]) / safe_base(
        row["ann_ret_research"])
    row["ann_vol_rel_gap"] = abs(row["ann_vol_gap"]) / safe_base(
        row["ann_vol_research"])
    row["sharpe_rel_gap"] = abs(row["sharpe_gap"]) / safe_base(
        row["sharpe_research"])
    row["profit_ratio_rel_gap"] = abs(row["profit_ratio_gap"]) / safe_base(
        row["profit_ratio_research"])
    row["turnover_rel_gap"] = abs(row["turnover_gap"]) / safe_base(
        row["turnover_research"])

    row["status"] = judge_inspect_status(row)
    return row


def run1(df, factor_name, return_name, pnl_method, cost_rate, holding_period):
    df1 = build_horizon_returns(df=df,
                                ret_col="nxt1_ret",
                                return_name=return_name,
                                holding_period=holding_period)

    profit_results, profit_daily, profit_month_return, profit_week_return = profitability(
        data=df1[['trade_time', factor_name, return_name]],
        factor_name=factor_name,
        return_name=return_name,
        cost_rate=cost_rate,
        max_pos=0,
        holding_period=holding_period,
        pnl_method=pnl_method,
    )
    spread_sequence, spread_results = quantile(
        data=df1[['trade_time', factor_name, return_name]],
        factor_name=factor_name,
        return_name=return_name,
    )

    pred_result = pred_metrics(
        data=df1[['trade_time', factor_name, return_name]],
        factor_name=factor_name,
        return_name=return_name)

    return {
        "summary": {
            "profit_results": profit_results,
            "spread_results": spread_results,
            "pred_result": pred_result,
        },
        "data": df1,
        "profit_daily": profit_daily,
        "profit_month_return": profit_month_return,
        "profit_week_return": profit_week_return,
        "spread_sequence": spread_sequence,
    }


def diagnostics(research_data, trader_data, factor_name, return_name,
                pnl_method, cost_rate, params):
    research_metrics = run1(df=research_data,
                            factor_name=factor_name,
                            return_name=return_name,
                            pnl_method=pnl_method,
                            cost_rate=cost_rate,
                            holding_period=params['horizon'])

    trader_metrics = run1(df=trader_data,
                          factor_name=factor_name,
                          return_name=return_name,
                          pnl_method=pnl_method,
                          cost_rate=cost_rate,
                          holding_period=params['horizon'])

    summary_compare = compare_summary(research_metrics, trader_metrics,
                                      factor_name)

    return {
        "summary": summary_compare,
        "research": research_metrics,
        "trader": trader_metrics,
    }
