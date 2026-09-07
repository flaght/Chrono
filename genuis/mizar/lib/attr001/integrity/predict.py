import pdb
import pandas as pd
import numpy as np
from chaosmind.timing.sirius0001.workflow import WorkFlow
from config.contract import INSTRUMENTS_CODES
from lib.attr001.ftd002 import *


def run1(factors_infos, params, code, symbol, task_id, factors_data):
    #features = [factor['formula'] for factor in factors_infos]
    workflow = WorkFlow(directory=params['model_path'],
                        code=code,
                        symbol=symbol,
                        task_id=task_id,
                        factors_infos=factors_infos,
                        softmax_temperature=params['softmax_temperature'],
                        min_open_signal_abs=params['min_open_signal_abs'],
                        method=params['method'],
                        win=params['win'])
    res = []
    pdb.set_trace()
    total_data1 = factors_data.dropna()
    all_trade_times = total_data1.index.get_level_values(
        'trade_time').unique().sort_values()
    for time in all_trade_times:
        print(time)
        rt = workflow.create_signals(trade_time=time, data=total_data1)
        res.append(rt)
    return pd.DataFrame(res)


def judge_prediction_status(summary):
    if (summary["pearson_corr"] >= 0.995 and summary["spearman_corr"] >= 0.995
            and summary["sign_match_ratio"] >= 0.99
            and summary["zero_cross_ratio"] <= 0.005):
        return "PASS"

    if (summary["pearson_corr"] >= 0.98 and summary["spearman_corr"] >= 0.98
            and summary["sign_match_ratio"] >= 0.95
            and summary["zero_cross_ratio"] <= 0.03):
        return "WARN"

    return "FAIL"

def netout_metrics(research_data, trader_data):
    feature = 'value'
    diff = research_data[feature] - trader_data[feature]
    abs_diff = diff.abs()
    sign_match = np.sign(research_data[feature]) == np.sign(
        trader_data[feature])
    zero_cross = (research_data[feature] * trader_data[feature]) < 0

    valid_count = int(len(research_data[feature])),
    pearson_corr = safe_corr(research_data[feature], trader_data[feature],
                             "pearson")
    spearman_corr = safe_corr(research_data[feature], trader_data[feature],
                              "spearman")
    sign_match_ratio = float(sign_match.mean())
    zero_cross_ratio = float(zero_cross.mean())
    p95_abs_diff = safe_quantile(abs_diff, 0.95)
    p99_abs_diff = safe_quantile(abs_diff, 0.99)
    summary = {
        'valid_count': valid_count,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'sign_match_ratio': sign_match_ratio,
        'zero_cross_ratio': zero_cross_ratio,
        'p95_abs_diff': p95_abs_diff,
        'p99_abs_diff': p99_abs_diff
    }

    summary["status"] = judge_prediction_status(summary)
    anomaly_threshold = safe_quantile(abs_diff, 0.95)

    anomalies = pd.DataFrame(abs_diff)
    anomalies['trade_time'] = research_data['trade_time']
    anomalies['code'] = research_data['code']
    anomalies['symbol'] = research_data['symbol']
    anomalies = anomalies[anomalies['value'] > anomaly_threshold].sort_values(
        by=['value'])

    return {"summary": summary, "anomalies": anomalies}


def diagnostics(factors_infos, params, instruments, task_id, research_data,
                trader_data):
    research_net_out = run1(factors_infos=factors_infos,
                            params=params,
                            code=INSTRUMENTS_CODES[instruments],
                            symbol='rb2610',
                            task_id=task_id,
                            factors_data=research_data)

    trader_net_out = run1(factors_infos=factors_infos,
                          params=params,
                          code=INSTRUMENTS_CODES[instruments],
                          symbol='rb2610',
                          task_id=task_id,
                          factors_data=trader_data)
    metrics_data = netout_metrics(research_data=research_net_out,
                                  trader_data=trader_net_out)
    metrics_data['research_net_out'] = research_net_out
    metrics_data['trader_net_out'] = trader_net_out
    return metrics_data
