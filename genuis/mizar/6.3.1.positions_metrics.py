import pdb, os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from kdutils.tactix import Tactix
from kdutils.macro2 import base_path, INSTRUMENTS_CODES
from lib.cux001 import FactorEvaluate1
from lib.bck002 import rebuild_executed_signal_for_eval, attach_position_labels
from lib.ret001 import create_chg, create_yields
from kdutils.data import fetch_local_market1


### 目标仓位和信号对比
def contrast_position_to_signal(position_data, signal_data):
    position_open = position_data[position_data['signal_type'] ==
                                  'open'].copy()
    position_check = position_open[[
        'trade_time', 'code', 'direction', 'numbers', 'pair_id'
    ]].merge(signal_data[['trade_time', 'code', 'signal', 'nxt1_ret_5h']],
             on=['trade_time', 'code'],
             how='left')
    print("position open vs raw signal diff:")
    print((position_check['direction'] != position_check['signal']).mean())
    print(pd.crosstab(position_check['direction'], position_check['signal']))
    return position_check


### 目标仓位和交易记录对吧
def contrast_position_to_trader(trader_data, position_data):
    trade_check = attach_position_labels(trader_data, position_data)
    print("trade records missing position rows:")
    print(trade_check['signal_type_position'].isna().mean())
    print(trade_check[trade_check['signal_type_position'].isna()].head(20))
    print("trade records by position signal_type:")
    print(trade_check['signal_type_position'].value_counts(dropna=False))
    return trade_check


### 信号和交易记录对比
def contrast_trader_to_signal1(signal_data, trade_check):
    open_trade = trade_check[trade_check['signal_type_position'] ==
                             'open'].copy()
    open_cmp = open_trade.merge(
        signal_data[["trade_time", "code", "signal", "nxt1_ret_5h"]],
        on=["trade_time", "code"],
        how="left")
    bad_open = open_cmp[open_cmp["direction"] != open_cmp["signal"]]
    print("open diff ratio:")
    print(len(bad_open) / len(open_cmp))
    print(pd.crosstab(open_cmp["direction"], open_cmp["signal"]))
    return open_cmp


### 可交易信号和交易记录对比
def contrast_trader_to_signal2(signal_data, trade_check):
    open_trade = trade_check[trade_check['signal_type_position'] ==
                             'open'].copy()
    signal_data_execute = signal_data[signal_data['signal'] != 0]
    open_cmp_execute = open_trade.merge(
        signal_data_execute[["trade_time", "code", "signal", "nxt1_ret_5h"]],
        on=["trade_time", "code"],
        how="right")
    bad_open = open_cmp_execute[open_cmp_execute["direction"] !=
                                open_cmp_execute["signal"]]
    print("open execute diff ratio:")
    print(len(bad_open) / len(open_cmp_execute))
    print(
        pd.crosstab(open_cmp_execute["direction"], open_cmp_execute["signal"]))

    return open_cmp_execute


### 可交易信号和交易记录对比 带开仓价 平仓价
def contrast_trader_to_signal3(signal_data, trade_check):
    open_leg_trade = trade_check[trade_check['signal_type_position'] ==
                                 'open'].copy()
    close_leg_trade = trade_check[trade_check['signal_type_position'] ==
                                  'close'].copy()

    trader_data1 = open_leg_trade[[
        'trade_time', 'pair_id', 'exec_price', 'code'
    ]].rename(columns={
        'exec_price': 'open_price'
    }).merge(close_leg_trade[['pair_id', 'exec_price', 'code',
                              'trade_time']].rename(columns={
                                  'exec_price': 'close_price',
                                  'trade_time': 'close_time'
                              }),
             on=['pair_id', 'code'])

    signal_data_execute = signal_data[signal_data['signal'] != 0]
    trader_execute = trader_data1.merge(
        signal_data_execute[["trade_time", "code", "signal", "nxt1_ret_5h"]],
        on=["trade_time", "code"],
        how="left")
    return trader_execute


def load_data(method, instruments, task_id, period, backtest_id,
              composite_method, composite_id, signal_method, signal_id):
    # signal_path = "/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/temp/model/113001/5/rl/signal/rl/1018806311332385/erband_signal/1002_test.feather"

    # position_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/temp/model/113001/5/rl/backtest/rl/1013836755991964/1018806311332385/erband_signal/1002_test/position_data.feather'

    # trader_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/temp/model/113001/5/rl/backtest/rl/1013836755991964/1018806311332385/erband_signal/1002_test/trade_records.feather'
    pdb.set_trace()
    signal_path = os.path.join(base_path, method, instruments, "temp", "model",
                               str(task_id), str(period),
                               "rl", "signal", composite_method,
                               str(composite_id), signal_method,
                               "{0}.feather".format(signal_id))
    position_path = os.path.join(base_path, method,
                                 instruments, "temp", "model", str(task_id),
                                 str(period), "rl", "backtest",
                                 composite_method, str(backtest_id),
                                 str(composite_id), signal_method,
                                 "{0}".format(signal_id),
                                 "position_data.feather")
    trader_path = os.path.join(base_path, method, instruments, "temp", "model",
                               str(task_id), str(period), "rl", "backtest",
                               composite_method, str(backtest_id),
                               str(composite_id), signal_method,
                               "{0}".format(signal_id),
                               "trade_records.feather")

    signal_data = pd.read_feather(signal_path)
    position_data = pd.read_feather(position_path)
    trader_data = pd.read_feather(trader_path)

    trader_data['trade_time'] = pd.to_datetime(
        trader_data['date'].astype(str) +
        trader_data['min_time'].astype(str).str.zfill(4),
        format='%Y-%m-%d%H%M')
    return signal_data, position_data, trader_data


###  信号数对比
def run1(method, instruments, task_id, period, backtest_id, composite_method,
         composite_id, signal_method, signal_id):

    pdb.set_trace()
    signal_data, position_data, trader_data = load_data(
        method=method,
        instruments=instruments,
        task_id=task_id,
        period=period,
        backtest_id=backtest_id,
        composite_method=composite_method,
        composite_id=composite_id,
        signal_method=signal_method,
        signal_id=signal_id)

    position_check = contrast_position_to_signal(position_data=position_data,
                                                 signal_data=signal_data)

    trade_check = contrast_position_to_trader(trader_data=trader_data,
                                              position_data=position_data)

    open_cmp = contrast_trader_to_signal1(signal_data=signal_data,
                                          trade_check=trade_check)

    open_cmp_execute = contrast_trader_to_signal2(signal_data=signal_data,
                                                  trade_check=trade_check)


### 收益率对比
def run2(method,
         instruments,
         task_id,
         period,
         backtest_id,
         composite_method,
         composite_id,
         signal_method,
         signal_id,
         tol=1e-6,
         top_n=5):
    signal_data, position_data, trader_data = load_data(
        method=method,
        instruments=instruments,
        task_id=task_id,
        period=period,
        backtest_id=backtest_id,
        composite_id=composite_id,
        composite_method=composite_method,
        signal_method=signal_method,
        signal_id=signal_id)

    trade_check = contrast_position_to_trader(trader_data=trader_data,
                                              position_data=position_data)

    trader_execute = contrast_trader_to_signal3(signal_data=signal_data,
                                                trade_check=trade_check)

    ## 读取行情
    start_date = trader_execute['trade_time'].min()
    end_date = trader_execute['trade_time'].max()
    non_factor_data = fetch_local_market1(
        base_path=os.environ['BAR_FUT_DIRS'],
        begin_date=start_date,
        end_date=end_date,
        method=None,
        codes=[INSTRUMENTS_CODES[instruments]])
    pcr_factor_data = fetch_local_market1(
        base_path=os.environ['BAR_FUT_DIRS'],
        begin_date=start_date,
        end_date=end_date,
        method='pcr',
        codes=[INSTRUMENTS_CODES[instruments]])

    ret_col = f"nxt1_ret_{period}h"
    non_col = f"non_nxt1_ret_{period}h"
    pcr_col = f"pcr_nxt1_ret_{period}h"

    res = []
    non_chg_data = create_chg(non_factor_data)
    non_df = create_yields(data=non_chg_data.copy(), horizon=period)
    non_df.name = non_col
    res.append(non_df)

    pcr_chg_data = create_chg(pcr_factor_data)
    pcr_df = create_yields(data=pcr_chg_data.copy(), horizon=period)
    pcr_df.name = pcr_col
    res.append(pcr_df)

    yields_data = pd.concat(res, axis=1)
    trader_execute = trader_execute.merge(yields_data,
                                          on=['trade_time', 'code'])
    trader_execute['execute_ret'] = np.log(trader_execute['close_price'] /
                                           trader_execute['open_price'])
    
    pairs = [(ret_col, non_col), (ret_col, pcr_col), (ret_col, "execute_ret"),
             (non_col, pcr_col), (non_col, "execute_ret"),
             (pcr_col, "execute_ret")]

    summary = []
    for left, right in pairs:
        valid = trader_execute[left].notna() & trader_execute[right].notna()
        diff = trader_execute.loc[valid, left] - trader_execute.loc[valid,
                                                                    right]
        abs_diff = diff.abs()
        summary.append({
            "left":
            left,
            "right":
            right,
            "rows":
            len(trader_execute),
            "valid_rows":
            int(valid.sum()),
            "missing_left":
            int(trader_execute[left].isna().sum()),  # 左侧列缺失数量
            "missing_right":
            int(trader_execute[right].isna().sum()),  # 右侧列缺失数量
            "mean_diff_bps":
            diff.mean() * 1e4,  # 平均有方向差异，单位 bps
            "mean_abs_diff_bps":
            abs_diff.mean() * 1e4,  # 平均绝对差异，单位 bps
            "p99_abs_diff_bps":
            abs_diff.quantile(0.99) * 1e4,  # 99% 分位的绝对差异。用来看大多数样本是否一致。
            "max_abs_diff_bps":
            abs_diff.max() * 1e4,
            "corr":
            trader_execute.loc[valid, [left, right]].corr().iloc[
                0, 1],  # 两列相关系数。接近 1 说明形状一致，但不代表数值完全一致
            "match_ratio":
            float((abs_diff <= tol
                   ).mean())  # 绝对差异小于等于 tol 的比例。你如果用 tol=1e-10，这个就是“几乎完全相等”的比例
        })

    # 每行差异，便于查 worst case
    for left, right in pairs:
        trader_execute[f"diff__{left}__{right}"] = trader_execute[
            left] - trader_execute[right]
        trader_execute[f"abs_diff_bps__{left}__{right}"] = trader_execute[
            f"diff__{left}__{right}"].abs() * 1e4

    summary = pd.DataFrame(summary)

    boundary_report = (trader_execute.groupby("trade_time")[[
        f"abs_diff_bps__{ret_col}__execute_ret",
        f"abs_diff_bps__{non_col}__execute_ret",
        f"abs_diff_bps__{pcr_col}__execute_ret"
    ]].agg(["count", "mean", "max"]).sort_index())

    worst = trader_execute.sort_values(f"abs_diff_bps__{ret_col}__execute_ret",
                                       ascending=False)

    key_summary = summary[summary[["left", "right"]].apply(
        tuple, axis=1).isin(pairs)][[
            "left", "right", "valid_rows", "missing_left", "missing_right",
            "mean_diff_bps", "mean_abs_diff_bps", "p99_abs_diff_bps",
            "max_abs_diff_bps", "corr", "match_ratio"
        ]].sort_values("mean_abs_diff_bps", ascending=False)

    main_boundary_col = f"abs_diff_bps__{ret_col}__execute_ret"
    key_boundary = (boundary_report[main_boundary_col].sort_values(
        "mean", ascending=False).head(top_n))

    main_abs_col = f"abs_diff_bps__{ret_col}__execute_ret"
    key_worst = trader_execute.sort_values(main_abs_col,
                                           ascending=False).head(top_n)

    show_cols = [
        "trade_time",
        "close_time",
        "pair_id",
        "code",
        "signal",
        "open_price",
        "close_price",
        ret_col,
        non_col,
        pcr_col,
        "execute_ret",
        f"abs_diff_bps__{ret_col}__{non_col}",
        f"abs_diff_bps__{ret_col}__{pcr_col}",
        f"abs_diff_bps__{ret_col}__execute_ret",
    ]
    show_cols = [c for c in show_cols if c in key_worst.columns]
    key_worst = key_worst[show_cols]

    print("\n[1] 关键总体对比")
    print(key_summary)

    print("\n[2] 差异最大的开仓分钟")
    print(key_boundary)

    print("\n[3] 差异最大的样本")
    print(key_worst)


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == "signal":
        run1(method=variant.method,
             instruments=variant.instruments,
             task_id=variant.task_id,
             period=variant.period,
             backtest_id=variant.backtest_id,
             composite_method=variant.composite_method,
             composite_id=variant.composite_id,
             signal_method=variant.signal_method,
             signal_id=variant.signal_id)
    elif variant.form == "metrics":
        run2(method=variant.method,
             instruments=variant.instruments,
             task_id=variant.task_id,
             period=variant.period,
             backtest_id=variant.backtest_id,
             composite_method=variant.composite_method,
             composite_id=variant.composite_id,
             signal_method=variant.signal_method,
             signal_id=variant.signal_id)
