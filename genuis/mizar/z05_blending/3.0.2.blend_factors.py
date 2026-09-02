## 筛选因子 构建等权组合
import os, copy, pdb
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv()

from lib.rl012.analysis import profitability, pred_metrics
from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.lsx001 import fetch_data  ## 加载原始数据
from lib.flp001 import load_data2 as load_metrics_data2


def load_data1(method, instruments, task_id, period):
    base_dirs = os.path.join(base_path, method, instruments, 'temp', 'model',
                             str(task_id), str(period), 'rl', 'data')

    train_data = pd.read_feather(os.path.join(base_dirs, "train_data.feather"))

    val_data = pd.read_feather(os.path.join(base_dirs, "val_data.feather"))

    return train_data, val_data


def load_data2(method, instruments, task_id, period):
    pdb.set_trace()
    base_dirs = os.path.join(base_path, method, instruments, 'temp', 'model',
                             str(task_id), str(period), 'rl', 'data')

    train_data = pd.read_feather(os.path.join(base_dirs, "train_data.feather"))

    val_data = pd.read_feather(os.path.join(base_dirs, "val_data.feather"))

    test_data = pd.read_feather(os.path.join(base_dirs, "test_data.feather"))
    return train_data, val_data, test_data


def load_metrics():
    metrics_data = pd.read_csv(
        "/workspace/worker/kdwk/Chrono/genuis/mizar/rb1.csv", index_col=0)
    metrics_data['abs_ic'] = metrics_data['ic_mean']
    return metrics_data


def load_metrics2(method, task_id, instruments, period, filename='choose.csv'):
    draft_data = load_metrics_data2(method=method,
                                    task_id=task_id,
                                    instruments=instruments,
                                    period=period,
                                    filename=filename)
    draft_data = draft_data.sort_values(by=['ann_sharpe'], ascending=False)[[
        'formula', 'ic_mean', 'ann_sharpe', 'calmar', 'pl_ratio'
    ]]
    draft_data['abs_ic'] = np.abs(draft_data['ic_mean'])
    return draft_data


def run1(method, instruments, task_id, period):
    metrics_data = load_metrics()
    metrics_data = metrics_data.sort_values(by=['abs_ic'], ascending=False)
    train_data, val_data = load_data1(method=method,
                                      instruments=instruments,
                                      task_id=task_id,
                                      period=period)
    total_data = pd.concat([train_data, val_data],
                           axis=0).sort_values(by=['trade_time', 'code'])
    selected = []
    selected_expr = []
    corr_threshold = 0.7

    for _, row in metrics_data.iterrows():
        expr = row["expression"]
        cur = total_data[["trade_time", expr]].dropna().copy()
        cur = cur.rename(columns={expr: "cur"})
        keep = True

        for kept_expr in selected_expr:
            kept = total_data[["trade_time", kept_expr]].dropna().copy()
            kept = kept.rename(columns={kept_expr: "kept"})
            tmp = cur.merge(kept, on="trade_time", how="inner")
            if len(tmp) < 50:
                continue

            corr = tmp["cur"].rolling(window=15, min_periods=5).corr(
                tmp["kept"]).replace([np.inf, -np.inf],
                                     np.nan).dropna().mean()
            print(expr, kept_expr, corr)
            if pd.notna(corr) and abs(corr) >= corr_threshold:
                keep = False
                break
        if keep:
            selected.append(row)
            selected_expr.append(expr)


def selected(method, instruments, task_id, period):
    pdb.set_trace()
    #metrics_data = load_metrics()
    metrics_data = load_metrics2(
        method=method,
        instruments=instruments,
        task_id=task_id,
        period=period,
        filename="chosen_pro.csv").rename(columns={'formula': 'expression'})
    metrics_data = metrics_data.sort_values(by=['abs_ic'], ascending=False)
    train_data, val_data = load_data1(method=method,
                                      instruments=instruments,
                                      task_id=task_id,
                                      period=period)
    total_data = pd.concat([train_data, val_data],
                           axis=0).sort_values(by=['trade_time', 'code'])

    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    selected_rows = {thr: [] for thr in thresholds}
    selected_exprs = {thr: [] for thr in thresholds}

    pdb.set_trace()
    factor_cache = {}
    for expr in metrics_data["expression"]:
        df = total_data[["trade_time", expr]].dropna().copy()
        df = df.rename(columns={expr: "value"})
        factor_cache[expr] = df

    corr_cache = {}

    def get_corr(expr1, expr2):
        key = tuple(sorted([expr1, expr2]))
        if key in corr_cache:
            return corr_cache[key]

        df1 = factor_cache[expr1].rename(columns={"value": "cur"})
        df2 = factor_cache[expr2].rename(columns={"value": "kept"})
        tmp = df1.merge(df2, on="trade_time", how="inner")

        if len(tmp) < 50:
            corr = np.nan
        else:
            corr = (tmp["cur"].rolling(window=15, min_periods=5).corr(
                tmp["kept"]).replace([np.inf, -np.inf],
                                     np.nan).dropna().mean())

        corr_cache[key] = corr
        return corr

    for _, row in metrics_data.iterrows():
        expr = row["expression"]

        for thr in thresholds:
            keep = True

            for kept_expr in selected_exprs[thr]:
                corr = get_corr(expr, kept_expr)
                print(expr, kept_expr, corr)
                if pd.notna(corr) and abs(corr) >= thr:
                    keep = False
                    break

            if keep:
                selected_rows[thr].append(row)
                selected_exprs[thr].append(expr)

    dt_path = os.path.join(base_path, method, instruments, 'temp', 'model',
                           str(task_id), str(period), 'rl', 'blend', 'corr')
    os.makedirs(dt_path, exist_ok=True)

    selected_metrics = {
        thr: pd.DataFrame(selected_rows[thr]).reset_index(drop=True)
        for thr in thresholds
    }
    pdb.set_trace()
    for thr in thresholds:
        filename = os.path.join(dt_path, "{0}.csv".format(str(thr * 10)))
        print(filename)
        selected_metrics[thr].to_csv(filename)


def create_evaluate(df, factor_names, pnl_ret_col, cost_rate, holding_period,
                    pnl_method, title_prefix, image_path):
    res1 = []
    res2 = []
    pdb.set_trace()
    for factor_name in factor_names:
        _, profit_daily, _, _ = profitability(
            data=df[['trade_time', factor_name, pnl_ret_col]],
            factor_name=factor_name,
            return_name=pnl_ret_col,
            cost_rate=cost_rate,
            max_pos=0,
            holding_period=holding_period,
            pnl_method=pnl_method,
        )
        net_nav = profit_daily['net_nav']
        net_nav.name = factor_name
        res1.append(net_nav)

        ic_sequence, _ = pred_metrics(
            data=df[['trade_time', factor_name, pnl_ret_col]],
            factor_name=factor_name,
            return_name=pnl_ret_col)
        s_ic = ic_sequence['s_ic']
        s_ic.name = factor_name
        res2.append(s_ic)

    profit_data = pd.concat(res1, axis=1)
    ic_data = pd.concat(res2, axis=1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 24))
    fig.suptitle(title_prefix, fontsize=16)

    ax1 = axes[0]

    ax1.plot(profit_data['corr_4'].index,
             profit_data['corr_4'].cumsum().values,
             label="Corr_4",
             color="orange",
             linewidth=1.8)
    ax1.plot(profit_data['corr_5'].index,
             profit_data['corr_5'].cumsum().values,
             label="Corr_5",
             color="royalblue",
             linewidth=1.8)
    ax1.plot(profit_data['corr_6'].index,
             profit_data['corr_6'].cumsum().values,
             label="Corr_6",
             color="purple",
             linewidth=1.8)
    ax1.plot(profit_data['corr_7'].index,
             profit_data['corr_7'].cumsum().values,
             label="Corr_7",
             color="seagreen",
             linewidth=1.8)
    ax1.plot(profit_data['corr_8'].index,
             profit_data['corr_8'].cumsum().values,
             label="Corr_8",
             color="goldenrod",
             linewidth=1.8)
    ax1.set_title("Net")
    ax1.set_ylabel("NAV")
    ax1.legend(loc="best")

    ax2 = axes[1]
    ax2.plot(ic_data['corr_4'].index,
             ic_data['corr_4'].cumsum().values,
             label="Corr_4",
             color="orange",
             linewidth=1.8)
    ax2.plot(ic_data['corr_5'].index,
             ic_data['corr_5'].cumsum().values,
             label="Corr_5",
             color="royalblue",
             linewidth=1.8)
    ax2.plot(ic_data['corr_6'].index,
             ic_data['corr_6'].cumsum().values,
             label="Corr_6",
             color="purple",
             linewidth=1.8)
    ax2.plot(ic_data['corr_7'].index,
             ic_data['corr_7'].cumsum().values,
             label="Corr_7",
             color="seagreen",
             linewidth=1.8)
    ax2.plot(ic_data['corr_8'].index,
             ic_data['corr_7'].cumsum().values,
             label="Corr_8",
             color="goldenrod",
             linewidth=1.8)
    ax2.set_ylabel("Cumulative IC")
    ax2.legend(loc="best")

    for ax in [ax1, ax2]:
        if ax.has_data():
            ax.tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    filename = os.path.join(image_path, "{}.png".format(title_prefix))
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(filename)
    plt.close(fig)


def composite(method, instruments, task_id, period):
    pdb.set_trace()
    train_data, val_data, test_data = load_data2(method=method,
                                                 instruments=instruments,
                                                 task_id=task_id,
                                                 period=period)

    dt_path = os.path.join(base_path, method, instruments, 'temp', 'model',
                           str(task_id), str(period), 'rl', 'blend', 'corr')

    file_path = Path(dt_path)
    for csv_file in file_path.rglob('*.csv'):
        corr_data = pd.read_csv(csv_file, index_col=0)
        name = csv_file.parts[-1].split('.')[0]
        train_data["corr_{0}".format(name)] = train_data[
            corr_data['expression'].to_list()].mean(axis=1)
        val_data["corr_{0}".format(name)] = val_data[
            corr_data['expression'].to_list()].mean(axis=1)
        test_data["corr_{0}".format(name)] = test_data[
            corr_data['expression'].to_list()].mean(axis=1)
    image_path = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl', 'blend',
                              'metrics')
    print(image_path)
    os.makedirs(image_path, exist_ok=True)
    pdb.set_trace()
    create_evaluate(
        df=train_data,
        factor_names=['corr_4', 'corr_5', 'corr_6', 'corr_8', 'corr_7'],
        pnl_ret_col='nxt1_ret_{0}h'.format(period),
        cost_rate=COST_MAPPING[INSTRUMENTS_CODES[instruments]],
        holding_period=period,
        pnl_method='points_norm',
        title_prefix="train",
        image_path=image_path)

    create_evaluate(
        df=val_data,
        factor_names=['corr_4', 'corr_5', 'corr_6', 'corr_8', 'corr_7'],
        pnl_ret_col='nxt1_ret_{0}h'.format(period),
        cost_rate=COST_MAPPING[INSTRUMENTS_CODES[instruments]],
        holding_period=period,
        pnl_method='points_norm',
        title_prefix="val",
        image_path=image_path)

    create_evaluate(
        df=test_data,
        factor_names=['corr_4', 'corr_5', 'corr_6', 'corr_8', 'corr_7'],
        pnl_ret_col='nxt1_ret_{0}h'.format(period),
        cost_rate=COST_MAPPING[INSTRUMENTS_CODES[instruments]],
        holding_period=period,
        pnl_method='points_norm',
        title_prefix="test",
        image_path=image_path)


if __name__ == '__main__':
    #selected(method='ricso2', instruments='rbb', task_id='113001', period=5)
    variant = Tactix().start()
    if variant.form == 'selected':
        selected(method=variant.method,
                 instruments=variant.instruments,
                 task_id=variant.task_id,
                 period=variant.period)
    elif variant.form == 'composite':
        composite(method=variant.method,
                  instruments=variant.instruments,
                  task_id=variant.task_id,
                  period=variant.period)
