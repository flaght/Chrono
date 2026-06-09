import os, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dotenv import load_dotenv

load_dotenv()


from lib.rl012.analysis import profitability, pred_metrics
from kdutils.tactix import Tactix
from kdutils.macro2 import *
from lib.uvx import * 

def _sanitize_frame(df: pd.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    bad_mask = ~np.isfinite(df[cols].to_numpy(dtype=np.float64))
    bad_count = int(bad_mask.sum())
    if bad_count > 0:
        print(f"[WARN] 数据中发现 {bad_count} 个 NaN/Inf，已填充为 0.0")
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def load_factors_data(method, instruments, task_id, period, features, ret_name):
    base_dirs = os.path.join(base_path, method, instruments, 'temp',
                             'model', str(task_id), str(period),
                               'rl', 'data')
    train_data = pd.read_feather(os.path.join(base_dirs,
                                              "train_data.feather"))
    
    val_data = pd.read_feather(os.path.join(base_dirs,
                                              "val_data.feather"))
    
    test_data = pd.read_feather(os.path.join(base_dirs,
                                              "test_data.feather"))
    
    train_data.rename(columns={ret_name: "nxt1_ret"},
                      inplace=True)

    val_data.rename(columns={ret_name: "nxt1_ret"},
                    inplace=True)
    
    test_data.rename(columns={ret_name: "nxt1_ret"},
                    inplace=True)
    
    
    train_data = train_data[['trade_time','code', 'nxt1_ret'] + features]
    val_data = val_data[['trade_time','code', 'nxt1_ret'] + features]
    test_data = test_data[['trade_time','code', 'nxt1_ret'] + features]
    
    train_data = train_data.sort_values('trade_time').reset_index(drop=True)
    val_data = val_data.sort_values('trade_time').reset_index(drop=True)
    test_data = test_data.sort_values('trade_time').reset_index(drop=True)
    
    train_data = _sanitize_frame(train_data, ['nxt1_ret'] + features)
    val_data = _sanitize_frame(val_data, ['nxt1_ret'] + features)
    test_data = _sanitize_frame(test_data, ['nxt1_ret'] + features)
    
    return train_data, val_data, test_data


def load_er_data(method, instruments, task_id, period, name):
    dirs = os.path.join(base_path, method, instruments, "temp", "model", str(task_id), str(period), "rl", "result", name, "metrics")
    train_er = pd.read_csv(os.path.join(dirs, "train_results.csv"))
    val_er = pd.read_csv(os.path.join(dirs, "val_results.csv"))
    test_er = pd.read_csv(os.path.join(dirs, "test_results.csv"))
    
    train_er['trade_time'] = pd.to_datetime(train_er['trade_time'])
    val_er['trade_time'] = pd.to_datetime(val_er['trade_time'])
    test_er['trade_time'] = pd.to_datetime(test_er['trade_time'])
    
    return train_er, val_er, test_er


def equal_weight(train_data, val_data, test_data, features, new_features):
    train_data['base_line'] = train_data[features].mean(axis=1)
    val_data['base_line'] = val_data[features].mean(axis=1)
    test_data['base_line'] = test_data[features].mean(axis=1)
    
    train_data['new_line'] = train_data[features + [new_features]].mean(axis=1)
    val_data['new_line'] = val_data[features + [new_features]].mean(axis=1)
    test_data['new_line'] = test_data[features + [new_features]].mean(axis=1)

    return train_data,val_data,test_data

def merge_data(train_data, val_data, test_data, train_er, val_er, test_er):
    train_data = train_data.merge(train_er[['trade_time','net_er_out','future_ret_h']], on=['trade_time'])
    val_data = val_data.merge(val_er[['trade_time','net_er_out','future_ret_h']], on=['trade_time'])
    test_data = test_data.merge(test_er[['trade_time','net_er_out','future_ret_h']], on=['trade_time'])
    return train_data, val_data, test_data
    
def run(method, instruments, task_id, period, name):
    ### IM
    features = [
        "MSUM(120,MDEMA(90,MCPS(90,'high')))",
        "MDEMA(120,MCPS(120,WMA(90,'twap')))",
        "MMAX(120,MDEMA(60,MCPS(90,'low')))",
    "MMASSI(120,MPRO(60,MVHF(10,'money')),MAPOSITIVE(10,'twap'))",
    "MDEMA(120,MCPS(120,MADecay(60,'twap')))",
    "MT3(120,MCPS(30,'close'))",
    "MA(60,RSI(120,MCPS(120,MA(60,'twap'))))",
    "MT3(120,MCPS(60,'high'))",
    "DELTA(90,MMIN(15,MHMA(90,DELTA(90,'close'))))/MDIFF(90,'close')",
    "MCPS(120,MT3(90,MMaxDiff(120,'twap')))",
    "MADecay(5,MMASSI(120,MT3(5,'corr_vwap_bid_size_0'),'twap'))",
    "MMeanRes(120,'corr_money_bid_size_0','smart_tick_in_pct')",
    "WMA(30,MMedian(90,'smart_tick_in_pct'))",
    "MMAX(15,MDPO(240,EMA(90,'smart_money_in_pct')))",
    "RSI(120,MCPS(120,EMA(120,'close')))",
    "MSUM(120,MDEMA(90,MCPS(90,'low')))",
    "MDIFF(90,MMeanRes(120,'corr_money_bid_size_0','smart_tick_in_pct'))",
    "MSUM(5,MADecay(10,MMedian(90,'smart_tick_in_pct')))",
    "MMedian(90,MADecay(10,MT3(5,'smart_tick_in_pct')))"
  ]
    new_features = "MMASSI(90,'twap',MADecay(15,'order_flow_imbanlace_1'))"
    
    name_uid = Params.create_tag({"features":features, "new_features":new_features})
    
    train_factors, val_factors, test_factors = load_factors_data(
        method=method, instruments=instruments, period=period, 
        task_id=task_id, features=features + [new_features],
        ret_name="nxt1_ret_{0}h".format(period))
    
    train_er, val_er, test_er = load_er_data(
        method=method, instruments=instruments, 
        task_id=task_id, period=period, name=name)
    
    train_factors, val_factors, test_factors = equal_weight(
        train_data=train_factors, val_data=val_factors, 
        test_data=test_factors, features=features, 
        new_features=new_features)
    
    train_factors, val_factors, test_factors = merge_data(train_data=train_factors, val_data=val_factors, 
               test_data=test_factors, 
               train_er=train_er, val_er=val_er, test_er=test_er)
    pdb.set_trace()
    image_path = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl', 'increment', name_uid)
    print(image_path)
    os.makedirs(image_path, exist_ok=True)
    create_evaluate(df=train_factors, factor_names=['net_er_out','base_line', 'new_line'], 
                    pnl_ret_col='future_ret_h',  
                    cost_rate=COST_MAPPING[INSTRUMENTS_CODES[instruments]], 
                    holding_period=period, 
                    pnl_method='points_norm',
                    title_prefix="train",
                    image_path=image_path)
    
    create_evaluate(df=val_factors, factor_names=['net_er_out','base_line', 'new_line'], 
                    pnl_ret_col='future_ret_h',  
                    cost_rate=COST_MAPPING[INSTRUMENTS_CODES[instruments]], 
                    holding_period=period, 
                    pnl_method='points_norm',
                    title_prefix="val",
                    image_path=image_path)
    
    create_evaluate(df=test_factors, factor_names=['net_er_out','base_line', 'new_line'], 
                    pnl_ret_col='future_ret_h',  
                    cost_rate=COST_MAPPING[INSTRUMENTS_CODES[instruments]], 
                    holding_period=period, 
                    pnl_method='points_norm',
                    title_prefix="test",
                    image_path=image_path)



def create_evaluate(df, factor_names, pnl_ret_col,  
                    cost_rate, holding_period, 
                    pnl_method, title_prefix,
                    image_path):
    res1 = []
    res2 = []
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
            factor_name=factor_name,return_name=pnl_ret_col)
        s_ic = ic_sequence['s_ic']
        s_ic.name = factor_name
        res2.append(s_ic)
        
    profit_data = pd.concat(res1,axis=1)
    ic_data = pd.concat(res2, axis=1)
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 24))
    fig.suptitle(title_prefix, fontsize=16)


    ax1 = axes[0]
    ax1.plot(profit_data['net_er_out'].index, profit_data['net_er_out'].cumsum().values, label="ER Out", color="orange", linewidth=1.8)
    ax1.plot(profit_data['base_line'].index, profit_data['base_line'].cumsum().values, label="Base Line", color="royalblue", linewidth=1.8)
    ax1.plot(profit_data['new_line'].index, profit_data['new_line'].cumsum().values, label="New Line", color="purple", linewidth=1.8)
    ax1.set_title("Net")
    ax1.set_ylabel("NAV")
    ax1.legend(loc="best")
    
    ax2 = axes[1]
    ax2.plot(ic_data['net_er_out'].index, ic_data['net_er_out'].cumsum().values, label="ER Out", color="orange", linewidth=1.8)
    ax2.plot(ic_data['base_line'].index, ic_data['base_line'].cumsum().values, label="Base Line", color="royalblue", linewidth=1.8)
    ax2.plot(ic_data['new_line'].index, ic_data['new_line'].cumsum().values, label="New Line", color="purple", linewidth=1.8)
    ax2.set_ylabel("Cumulative IC")
    ax2.legend(loc="best")
    
    for ax in [ax1, ax2]:
        if ax.has_data():
            ax.tick_params(axis="x", rotation=30)
            
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    
    filename = os.path.join(image_path, "{}.png".format(title_prefix))
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    
    
    
    
if __name__ == '__main__':
    variant = Tactix().start()
    run(method=variant.method, instruments=variant.instruments, 
        task_id=variant.task_id, 
        period=variant.period,
        name=variant.name)