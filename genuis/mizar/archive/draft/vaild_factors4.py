import pdb
import pandas as pd
import numpy as np

from lib.cux001 import FactorEvaluate1

from lib.rl012.analysis import profitability, pred_metrics

def add_canary(df, rho=0.6, seed=42):
    rng = np.random.default_rng(seed)
    y = df['nxt1_ret_15h'].astype(float).to_numpy()

    mu = np.nanmean(y)
    sd = np.nanstd(y) + 1e-8

    zy = (y - mu) / sd
    e = rng.normal(size=len(df))
    x = rho * zy + np.sqrt(1 - rho**2) * e 
    df[f"CANARY_GOOD_{int(rho*100):02d}"] = x
    return df




def fit_z(train_s: pd.Series):
    mu = float(np.nanmean(train_s))
    sd = float(np.nanstd(train_s) + 1e-8)
    return mu, sd

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
    # d = df[["trade_time", sig_col, "future_ret_h"]].copy()
    # d = d.rename(columns={sig_col: "net_er_out"})
    # evaluate1 = FactorEvaluate1(factor_data=d.copy(),
    #                             factor_name='net_er_out',
    #                             ret_name='future_ret_h',
    #                             roll_win=15,
    #                             fee=0.000,
    #                             scale_method='raw',
    #                             expression='expression',
    #                             resampling_win=15)
    # stats_dt1 = evaluate1.run()
    # return stats_dt1
    
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


train_net_er = pd.read_csv("./records/cicso1/ims/temp/model/200037/15/rl/result/1087395932985978/metrics/train_results.csv")
train_factors = pd.read_feather("./records/cicso1/ims/temp/model/200037/15/rl/data/train_data.feather")

val_net_er = pd.read_csv("./records/cicso1/ims/temp/model/200037/15/rl/result/1087395932985978/metrics/val_results.csv")
val_factors = pd.read_feather("./records/cicso1/ims/temp/model/200037/15/rl/data/val_data.feather")

test_net_er = pd.read_csv("./records/cicso1/ims/temp/model/200037/15/rl/result/1087395932985978/metrics/test_results.csv")
test_factors = pd.read_feather("./records/cicso1/ims/temp/model/200037/15/rl/data/test_data.feather")

train_factors = add_canary(train_factors, rho=0.6)
val_factors = add_canary(val_factors, rho=0.6)
test_factors = add_canary(test_factors, rho=0.6)




val_net_er = val_net_er.reset_index(drop=True)
val_factors = val_factors.reset_index(drop=True)
test_net_er = test_net_er.reset_index(drop=True)
test_factors = test_factors.reset_index(drop=True)

pdb.set_trace()
# factor_name = "MRANK(5, DELTA(5, ADDED('price_imbalance_0', 'depth_imbalance_4')))"

factor_name = "CANARY_GOOD_60"

train_mu_net, train_sd_net = fit_z(train_net_er["net_er_out"])
train_mu_factor, train_sd_factor = fit_z(train_factors[factor_name])


base_abs_cap = float(np.nanquantile(np.abs(train_net_er["net_er_out"]), 0.995))
train_zb = apply_z(train_net_er["net_er_out"], train_mu_net, train_sd_net)
train_zc = apply_z(train_factors[factor_name], train_mu_factor, train_sd_factor)

base_val = eval_one_signal(val_net_er.rename(columns={"net_er_out": "base_sig"}), "base_sig")
base_test = eval_one_signal(test_net_er.rename(columns={"net_er_out": "base_sig"}), "base_sig")

lambdas = [0.05, 0.10, 0.20]
rows = []
# pdb.set_trace()
for lam in lambdas:
    # 每个lambda都在训练集算一次 mix 的标准差
    train_mix_z = (1.0 - lam) * train_zb + lam * train_zc
    mix_std_train = float(np.nanstd(train_mix_z) + 1e-8)

    val_net_er[f"mix_{lam}"] = build_mix_signal(
        val_net_er["net_er_out"], val_factors[factor_name],
        train_mu_net, train_sd_net, train_mu_factor, train_sd_factor,
        lam, mix_std_train, base_abs_cap
    )
    test_net_er[f"mix_{lam}"] = build_mix_signal(
        test_net_er["net_er_out"], test_factors[factor_name],
        train_mu_net, train_sd_net, train_mu_factor, train_sd_factor,
        lam, mix_std_train, base_abs_cap
    )

    m_val = eval_one_signal(val_net_er, f"mix_{lam}")
    m_test = eval_one_signal(test_net_er, f"mix_{lam}")
    rows.append({
        "lambda": lam,
        "d_val_ann_ret": m_val["ann_ret"] - base_val["ann_ret"],
        "d_val_rank_ic": m_val["rank_ic"] - base_val["rank_ic"],
        "d_test_ann_ret": m_test["ann_ret"] - base_test["ann_ret"],
        "d_test_rank_ic": m_test["rank_ic"] - base_test["rank_ic"],
        "test_turnover_ratio": m_test["turnover"] / (base_test["turnover"] + 1e-8),
        "d_test_calmar": m_test["calmar"] - base_test["calmar"],
    })
    
inc = pd.DataFrame(rows)
pdb.set_trace()
print(inc)
