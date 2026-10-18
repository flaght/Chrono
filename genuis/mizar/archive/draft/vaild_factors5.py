import pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from lib.uvx import * 
from lib.rl012.analysis import profitability, pred_metrics


def load_er_data(method, instruments, task_id, period, name='1087395932985978'):
    dirs = os.path.join(base_path, method, instruments, "temp", "model", str(task_id), str(period), "rl", "result", name, "metrics")
    train_er = pd.read_csv(os.path.join(dirs, "train_results.csv"))
    val_er = pd.read_csv(os.path.join(dirs, "val_results.csv"))
    test_er = pd.read_csv(os.path.join(dirs, "test_results.csv"))
    return train_er, val_er, test_er
    
    
def load_factor_data(method, instruments, task_id, period):
    dirs = os.path.join(base_path, method, instruments, "temp", "model", str(task_id), str(period), "rl", "data")
    train_factors = pd.read_feather(os.path.join(dirs, "train_data.feather"))
    val_factors = pd.read_feather(os.path.join(dirs, "val_data.feather"))
    test_factors = pd.read_feather(os.path.join(dirs, "test_data.feather"))
    return train_factors,val_factors,test_factors


def load_results():
    select_factors = pd.read_csv("/workspace/worker/kdwk/Chrono/genuis/mizar/ims1.csv")
    return select_factors


def fit_z(train_s: pd.Series):
    mu = float(np.nanmean(train_s))
    sd = float(np.nanstd(train_s) + 1e-8)
    return mu, sd

def apply_z(x: pd.Series, mu: float, sd: float):
    return (x - mu) / sd

def build_mix_signal(base_s, cand_s, mu_b, sd_b, mu_c, sd_c, lam, mix_std_train, base_abs_cap):
    # 标准化
    zb = apply_z(base_s, mu_b, sd_b)
    zc = apply_z(cand_s, mu_c, sd_c)

    # 保守混合
    mix_z = (1.0 - lam) * zb + lam * zc

    # 保幅1：按训练集混合标准差归一，防止整体放大
    mix_z = mix_z / (mix_std_train + 1e-8)

    # 回到 base 原始尺度
    mix_raw = mix_z * sd_b + mu_b

    # 保幅2：裁剪到 base 训练集幅度
    mix_raw = mix_raw.clip(-base_abs_cap, base_abs_cap)
    return mix_raw



def eval_one_signal(df: pd.DataFrame, sig_col: str, cost=0.000023, hp=15):
    d = df[["trade_time", sig_col, "future_ret_h"]].copy()
    d = d.rename(columns={sig_col: "net_er_out"})
    prof, _, _, _ = profitability(
        data=d[["trade_time", "net_er_out", "future_ret_h"]],
        factor_name="net_er_out",
        return_name="future_ret_h",
        cost_rate=cost,
        max_pos=0,
        holding_period=hp,
        pnl_method="points_norm",
    )
    _, pred = pred_metrics(
        data=d[["trade_time", "net_er_out", "future_ret_h"]],
        factor_name="net_er_out",
        return_name="future_ret_h",
    )
    return {
        "ann_ret": float(prof["ann_ret"]),
        "calmar": float(prof["calmar"]),
        "turnover": float(prof["turnover"]),
        "rank_ic": float(pred["total_rank_ic"]),
    }
    
    
def calc(train_er, val_er, test_er, 
         train_factors, val_factors, test_factors, 
         factor_names):
    lambdas = [0.05, 0.10, 0.20]
    train_mu_net, train_sd_net = fit_z(train_er["net_er_out"])
    base_abs_cap = float(np.nanquantile(np.abs(train_er["net_er_out"]), 0.995))
    train_zb = apply_z(train_er["net_er_out"], train_mu_net, train_sd_net)
    base_val = eval_one_signal(val_er.rename(columns={"net_er_out": "base_sig"}), "base_sig")
    base_test = eval_one_signal(test_er.rename(columns={"net_er_out": "base_sig"}), "base_sig")
    pdb.set_trace()
    rows = []
    for factor_name in factor_names:
        train_mu_factor, train_sd_factor = fit_z(train_factors[factor_name])
        train_zc = apply_z(train_factors[factor_name], train_mu_factor, train_sd_factor)
        for lam in lambdas:
            print(factor_name, lam)
            train_mix_z = (1.0 - lam) * train_zb + lam * train_zc
            mix_std_train = float(np.nanstd(train_mix_z) + 1e-8)
            
            
            val_net_er1 = val_er.copy()
            test_net_er1 = test_er.copy()
            val_net_er1[f"mix_{lam}"] = build_mix_signal(
                val_net_er1["net_er_out"], val_factors[factor_name].copy(),
                train_mu_net, train_sd_net, train_mu_factor, train_sd_factor,
                lam, mix_std_train, base_abs_cap)
            
            test_net_er1[f"mix_{lam}"] = build_mix_signal(
                test_net_er1["net_er_out"], test_factors[factor_name].copy(),
                train_mu_net, train_sd_net, train_mu_factor, train_sd_factor,
                lam, mix_std_train, base_abs_cap)
            
            m_val = eval_one_signal(val_net_er1, f"mix_{lam}")
            m_test = eval_one_signal(test_net_er1, f"mix_{lam}")
            
            rows.append({
                "name":factor_name,
                "lambda": lam,
                "d_val_ann_ret": m_val["ann_ret"] - base_val["ann_ret"],
                "d_val_rank_ic": m_val["rank_ic"] - base_val["rank_ic"],
                "d_test_ann_ret": m_test["ann_ret"] - base_test["ann_ret"],
                "d_test_rank_ic": m_test["rank_ic"] - base_test["rank_ic"],
                "test_turnover_ratio": m_test["turnover"] / (base_test["turnover"] + 1e-8),
                "d_test_calmar": m_test["calmar"] - base_test["calmar"]})
            
    pdb.set_trace()
    print() 
    
pdb.set_trace()
method = "cicso1"
instruments = "ims"
task_id = "200037"
period = "15"

train_er, val_er, test_er = load_er_data(method=method, instruments=instruments, task_id=task_id, period=period)
train_factors,val_factors,test_factors = load_factor_data(method=method, instruments=instruments, task_id=task_id, period=period)
select_factors = load_results()
pdb.set_trace()
calc(train_er=train_er, val_er=val_er, test_er=test_er, 
         train_factors=train_factors, val_factors=val_factors, 
         test_factors=test_factors, 
         factor_names= select_factors['expression'].tolist())
    