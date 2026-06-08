import pdb
import pandas as pd
from lib.cux003 import FactorEvaluate1


def judge_performance_status(row):
    total_ic_gap = abs(row["total_ic_gap"])
    ic_mean_gap = abs(row["ic_mean_gap"])
    sharpe_gap = abs(row["sharpe1_gap"])
    total_ret_gap = abs(row["total_ret_gap"])

    if (total_ic_gap <= 0.01 and ic_mean_gap <= 0.01 and sharpe_gap <= 0.05
            and total_ret_gap <= 0.002):
        return "PASS"

    if (total_ic_gap <= 0.03 and ic_mean_gap <= 0.03 and sharpe_gap <= 0.15
            and total_ret_gap <= 0.005):
        return "WARN"

    if (total_ic_gap <= 0.05 and ic_mean_gap <= 0.05 and sharpe_gap <= 0.30
            and total_ret_gap <= 0.01):
        return "FAIL_EDGE"

    return "FAIL"


def compare_factor(research_eval, trader_eval):
    row = {
        "name": research_eval["name"],
        "total_ret_research": research_eval.get("total_ret"),
        "total_ret_trader": trader_eval.get("total_ret"),
        "avg_ret_research": research_eval.get("avg_ret"),
        "avg_ret_trader": trader_eval.get("avg_ret"),
        "max_dd_research": research_eval.get("max_dd"),
        "max_dd_trader": trader_eval.get("max_dd"),
        "calmar_research": research_eval.get("calmar"),
        "calmar_trader": trader_eval.get("calmar"),
        "sharpe1_research": research_eval.get("sharpe1"),
        "sharpe1_trader": trader_eval.get("sharpe1"),
        "sharpe2_research": research_eval.get("sharpe2"),
        "sharpe2_trader": trader_eval.get("sharpe2"),
        "turnover_research": research_eval.get("turnover"),
        "turnover_trader": trader_eval.get("turnover"),
        "win_rate_research": research_eval.get("win_rate"),
        "win_rate_trader": trader_eval.get("win_rate"),
        "profit_ratio_research": research_eval.get("profit_ratio"),
        "profit_ratio_trader": trader_eval.get("profit_ratio"),
        "total_ic_research": research_eval.get("total_ic"),
        "total_ic_trader": trader_eval.get("total_ic"),
        "ic_mean_research": research_eval.get("ic_mean"),
        "ic_mean_trader": trader_eval.get("ic_mean"),
        "ic_std_research": research_eval.get("ic_std"),
        "ic_std_trader": trader_eval.get("ic_std"),
        "ic_ir_research": research_eval.get("ic_ir"),
        "ic_ir_trader": trader_eval.get("ic_ir"),
        "factor_autocorr_research": research_eval.get("factor_autocorr"),
        "factor_autocorr_trader": trader_eval.get("factor_autocorr"),
        "ret_autocorr_research": research_eval.get("ret_autocorr"),
        "ret_autocorr_trader": trader_eval.get("ret_autocorr"),
    }

    for metric in [
            "total_ret", "avg_ret", "max_dd", "calmar", "sharpe1", "sharpe2",
            "turnover", "win_rate", "profit_ratio", "total_ic", "ic_mean",
            "ic_std", "ic_ir", "factor_autocorr", "ret_autocorr"
    ]:
        row[f"{metric}_gap"] = row[f"{metric}_trader"] - row[
            f"{metric}_research"]

    row["status"] = judge_performance_status(row)
    return row


def factor_evaluate(factors_data,
                    factor_name,
                    horizon,
                    roll_win=15,
                    fee=0.0,
                    scale_method='raw'):
    evaluate1 = FactorEvaluate1(factor_data=factors_data,
                                factor_name=factor_name,
                                ret_name="nxt1_ret",
                                roll_win=roll_win,
                                fee=fee,
                                scale_method=scale_method,
                                expression=factor_name,
                                resampling_win=horizon)
    dt2 = evaluate1.run()
    dt2['name'] = factor_name
    return dt2


def run_evaluate(research_data, trader_data, factors_infos, params):
    rows = []
    research_results = {}
    trader_results = {}
    for factor in factors_infos:
        factor_name = factor["formula"]
        research_eval = factor_evaluate(factors_data=research_data,
                                        factor_name=factor_name,
                                        horizon=params['horizon'])
        trader_eval = factor_evaluate(factors_data=trader_data,
                                      factor_name=factor_name,
                                      horizon=params['horizon'])
        research_results[factor_name] = research_eval
        trader_results[factor_name] = trader_eval
        rows.append(compare_factor(research_eval, trader_eval))
    metrics = pd.DataFrame(rows)
    summary = {
        "factor_count":
        int(len(metrics)),
        "status_count":
        metrics["status"].value_counts().to_dict()
        if not metrics.empty else {},
        "performance_fail_count":
        int(metrics["status"].isin(["FAIL", "FAIL_EDGE"]).sum())
        if not metrics.empty else 0,
        "total_ic_gap_abs_mean":
        float(metrics["total_ic_gap"].abs().mean())
        if not metrics.empty else float("nan"),
        "ic_mean_gap_abs_mean":
        float(metrics["ic_mean_gap"].abs().mean())
        if not metrics.empty else float("nan"),
        "total_ret_gap_abs_mean":
        float(metrics["total_ret_gap"].abs().mean())
        if not metrics.empty else float("nan"),
        "sharpe1_gap_abs_mean":
        float(metrics["sharpe1_gap"].abs().mean())
        if not metrics.empty else float("nan"),
    }

    return {
        "summary": summary,
        "metrics": metrics,
        "research_results": research_results,
        "trader_results": trader_results,
    }


def diagnostics(research_data, trader_data, factors_infos, params):
    ### 处理数据
    research_data1 = research_data.dropna().set_index(['trade_time', 'code'])
    trader_data1 = trader_data.dropna().set_index(['trade_time', 'code'])

    comm_index = research_data1.index.intersection(trader_data1.index)
    research_data1 = research_data1.loc[comm_index].reset_index().sort_values(
        by=['trade_time', 'code'])
    trader_data1 = trader_data1.loc[comm_index].reset_index().sort_values(
        by=['trade_time', 'code'])

    metrics_results = run_evaluate(research_data=research_data1,
                                   trader_data=trader_data1,
                                   factors_infos=factors_infos,
                                   params=params)
    return metrics_results
