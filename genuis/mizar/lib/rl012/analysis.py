import pandas as pd
import numpy as np
from collections import deque
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pdb

def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    x1 = x[mask]
    y1 = y[mask]
    if float(x1.std(ddof=0)) == 0.0 or float(y1.std(ddof=0)) == 0.0:
        return 0.0
    v = x1.corr(y1, method=method)
    return float(v) if pd.notna(v) else 0.0


"""
时序 Pearson（不滚动）
测整体线性关系：分数高时，未来收益幅度是否整体更大。

时序 Spearman（不滚动）
测整体排序关系：分数排名高的时点，未来收益排名是否整体更高。

滚动 Pearson
看线性关系是否随时间变化，识别“这几个月有效、那几个月失效”。

滚动 Spearman
看排序能力是否随时间变化，识别“模型择时排序是否稳定”。
"""
def pred_metrics(data:pd.DataFrame, factor_name: str, return_name: str):
    data = data.copy()
    data['trade_date'] = pd.to_datetime(data['trade_time']).dt.date
    res = []
    for k, g in data.groupby("trade_date"):
        ## 强弱性，置信度越高，收益绝对值也越大
        p_ic = safe_corr(g[factor_name], g[return_name], method="pearson")
        ## 择时排序能力
        s_ic = safe_corr(g[factor_name], g[return_name], method="spearman")
        res.append({'trade_date':k, 'p_ic':p_ic, 's_ic':s_ic})
    ic_sequence = pd.DataFrame(res).set_index('trade_date')
    person_ic = safe_corr(data[factor_name], data[return_name], method="pearson")
    rank_ic = safe_corr(data[factor_name], data[return_name], method="spearman")
    return ic_sequence, {'total_person_ic':person_ic, 
                         'total_rank_ic':rank_ic,
                         'person_ic_mean':ic_sequence['p_ic'].mean(),
                         'rank_ic_mean':ic_sequence['s_ic'].mean(),
                         'person_icir':ic_sequence['p_ic'].mean()/ic_sequence['p_ic'].std(),
                         'rank_icir':ic_sequence['s_ic'].mean()/ic_sequence['s_ic'].std()
                         }
                         
    
def quantile(data:pd.DataFrame, factor_name: str, return_name: str):
    data = data.copy()
    data['trade_date'] = pd.to_datetime(data['trade_time']).dt.date
    data = data[data[factor_name] != 0].copy()
    data["signed_ret"] = np.sign(data[factor_name]) * data[return_name] # 方向收益：模型真正交易到的收益
    
    spread_sequence = (
        data.groupby("trade_date").apply(lambda g: g.loc[g[factor_name] >= g[factor_name].quantile(0.9), return_name].mean()
                     - g.loc[g[factor_name] <= g[factor_name].quantile(0.1), return_name].mean()))
    q_hi = float(data[factor_name].quantile(0.9))
    q_lo = float(data[factor_name].quantile(0.1))
    high_mean = float(data.loc[data[factor_name] >= q_hi, "signed_ret"].mean())
    low_mean = float(data.loc[data[factor_name] <= q_lo, "signed_ret"].mean())
    spread = high_mean - low_mean
    return spread_sequence, {'total_spread':spread, 'spread_mean':spread_sequence.mean()}
    
def profitability(data: pd.DataFrame, factor_name:str,
                    return_name:str, cost_rate:float,
                    max_pos: int, holding_period:int = 15,
                    annual_days:int = 250, 
                    pnl_method:str ='raw'):
    
    hp = max(int(holding_period), 1)
    data = data.copy().set_index('trade_time')
    action_arr = data[factor_name].to_numpy(dtype=np.float64)
    ret_arr = data[return_name].to_numpy(dtype=np.float64)
    n = len(ret_arr)
    pnl_method = str(pnl_method).strip().lower()
    if pnl_method not in {"raw", "normalized", "points_norm"}:
        raise ValueError("pnl_method must be one of {'raw', 'normalized', 'points_norm'}")
    # use_phase_avg 与 pnl_method 解耦，避免“口径选择”改变“持仓聚合机制”。
    # 当前固定规则：holding_period>1 时启用 phase_avg。
    use_phase_avg = hp > 1
    
    
    def _run_sim(entry_mask: Optional[np.ndarray]):
        # 固定持有期 + 可叠加 + 净持仓限幅口径：
        # 每条信号从 t 持有到 t+hp-1，t+hp 自动到期。
        # max_position > 0 时按净持仓限幅；max_position == 0 时不限制。
        pos_arr = np.zeros(n, dtype=np.float64)
        active = deque()
        active_net = 0.0
        limit_enabled = bool(max_pos > 0.0)
        i = 0
        while i < n:
            while active and active[0][0] <= i:
                _, expired_size = active.popleft()
                active_net -= float(expired_size)

            allow_entry = True if entry_mask is None else bool(entry_mask[i])
            sig = float(action_arr[i]) if np.isfinite(action_arr[i]) else 0.0
            if allow_entry and sig != 0.0:
                if limit_enabled:
                    target_net = float(np.clip(active_net + sig, -max_pos, max_pos))
                    add_size = target_net - active_net
                else:
                    add_size = sig
            
                if add_size != 0.0:
                    active.append((i + hp, float(add_size)))
                    active_net += float(add_size)
            pos_arr[i] = active_net
            i += 1
        
        turnover_arr = np.abs(pos_arr - np.r_[0.0, pos_arr[:-1]])
        gross_ret_arr = pos_arr * ret_arr
        # fee_cost_arr = float(cost_rate) * turnover_arr
        # net_ret_arr = gross_ret_arr - fee_cost_arr
    
        if pnl_method == 'normalized':
            # normalized + phase_avg 下不再额外除 hp，避免双重缩放
            norm_scale = 1.0 if use_phase_avg else float(hp)
            p_gross = gross_ret_arr / norm_scale
            p_fee = float(cost_rate) * (turnover_arr / norm_scale)
        else:
            p_gross = gross_ret_arr
            p_fee = float(cost_rate) * turnover_arr
        
        p_net = p_gross - p_fee
        return pos_arr, turnover_arr, p_gross, p_net

    def _run_points_norm():
        # 点数归一化口径（名义资金）：
        # 逐笔在到期时确认盈亏，并用 N_pairs(=holding_period) 做归一化。
        # 这里 return_name 直接使用外部已计算好的“入场时对应持有期收益”。
        pos_arr = np.zeros(n, dtype=np.float64)
        realized_arr = np.zeros(n, dtype=np.float64)
        active = deque()  # (exit_idx, size, trade_ret)
        active_net = 0.0
        limit_enabled = bool(max_pos > 0.0)

        i = 0
        while i < n:
            while active and active[0][0] <= i:
                _, expired_size, trade_ret = active.popleft()
                active_net -= float(expired_size)
                if np.isfinite(trade_ret):
                    realized_arr[i] += float(expired_size) * float(trade_ret)

            sig = float(action_arr[i]) if np.isfinite(action_arr[i]) else 0.0
            if sig != 0.0:
                if limit_enabled:
                    target_net = float(np.clip(active_net + sig, -max_pos, max_pos))
                    add_size = target_net - active_net
                else:
                    add_size = sig

                if add_size != 0.0:
                    trade_ret = float(ret_arr[i]) if np.isfinite(ret_arr[i]) else 0.0
                    active.append((i + hp, float(add_size), trade_ret))
                    active_net += float(add_size)

            pos_arr[i] = active_net
            i += 1

        turnover_arr = np.abs(pos_arr - np.r_[0.0, pos_arr[:-1]]) * 0.5
        
        n_pairs = float(max(hp, 1))
        p_gross = realized_arr / n_pairs
        p_fee = float(cost_rate) * (turnover_arr / n_pairs)
        p_net = p_gross - p_fee
        return pos_arr, turnover_arr, p_gross, p_net
            
    use_phase_avg_effective = bool(use_phase_avg and pnl_method != "points_norm")
    #print("pnl_method:{0},use_phase_avg:{1}".format(pnl_method, use_phase_avg_effective))
    if pnl_method == "points_norm":
        pos_arr, turnover_arr, gross_ret_arr, net_ret_arr = _run_points_norm()
    elif use_phase_avg_effective:
        # phase_avg:
        # 将 t % hp 的 hp 个相位子策略分别模拟，再对收益/仓位取均值。
        # 相比“直接除以 hp”，这是更稳的重叠持仓近似。
        # 但它本质仍是资金切片近似，不等同逐笔成交回测。
        phase_pos = []
        phase_turnover = []
        phase_gross = []
        phase_net = []
        idx_arr = np.arange(n, dtype=np.int64)
        for phase in range(hp):
            mask = (idx_arr % hp) == phase
            p_pos, p_turnover, p_gross, p_net = _run_sim(entry_mask=mask)
            phase_pos.append(p_pos)
            phase_turnover.append(p_turnover)
            phase_gross.append(p_gross)
            phase_net.append(p_net)
        pos_arr = np.mean(np.vstack(phase_pos), axis=0)
        turnover_arr = np.mean(np.vstack(phase_turnover), axis=0)
        gross_ret_arr = np.mean(np.vstack(phase_gross), axis=0)
        net_ret_arr = np.mean(np.vstack(phase_net), axis=0)
    else:
        pos_arr, turnover_arr, gross_ret_arr, net_ret_arr = _run_sim(entry_mask=None)
    
    
    
    data.index = pd.to_datetime(data.index)
    pos = pd.Series(pos_arr, index=data.index, dtype=np.float64)
    turnover = pd.Series(turnover_arr, index=data.index, dtype=np.float64)
    gross_ret = pd.Series(gross_ret_arr, index=data.index, dtype=np.float64)
    net_nav = pd.Series(net_ret_arr, index=data.index, dtype=np.float64)
    nav = np.exp(net_nav.cumsum())
    net_simple = np.expm1(net_nav)
    win_rate_seq = net_simple.resample("1D").apply(lambda s: (s > 0).mean() if len(s) > 0 else np.nan)
    profit_ratio_seq = net_simple.resample("1D").apply(
        lambda s: (s[s > 0].sum() / abs(s[s < 0].sum())if abs(s[s < 0].sum()) > 1e-12 else np.nan))
    
    
    ## 换算日频
    metrics_data = pd.DataFrame(
        {
         'turnover':turnover,
         'gross_nav':gross_ret,
         'net_nav':net_nav,
         'nav':nav
         },index=data.index)
    
    metrics_data = metrics_data.copy()
    metrics_data.index = pd.to_datetime(metrics_data.index)
    metrics_data = metrics_data.sort_index()
    

    daily = metrics_data.resample("1D").agg(
        {
            "turnover": "mean",
            "gross_nav": "sum",
            "net_nav": "sum",
        }
    )
    # daily['gross_nav'] = daily['gross_nav'] /holding_period
    # daily['net_nav'] = daily['net_nav'] /holding_period
    daily.index.name = "trade_date"
    
    daily = daily.dropna(subset=["turnover", "gross_nav", "net_nav"])
    daily["equity"] = np.exp(daily["net_nav"].cumsum())
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    # 向后兼容：maxdd列保存“回撤序列(<=0)”
    daily["maxdd"] = daily["drawdown"]
    
    daily["win_rate"] = win_rate_seq
    daily["profit_ratio"] = profit_ratio_seq
    ## 绩效评估
    mean_daily_log = float(daily["net_nav"].mean())
    ann_ret = float(np.exp(mean_daily_log * float(annual_days)) - 1.0)
    
    ann_vol = daily["net_nav"].std(ddof=0) * np.sqrt(annual_days)
    maxdd = float(-daily["maxdd"].min()) if len(daily) > 0 else 0.0
    calmar = ann_ret / maxdd if maxdd > 1e-12 else 0.0
    sharpe = (
        mean_daily_log / float(daily["net_nav"].std(ddof=0)) * np.sqrt(annual_days)
        if (len(daily) > 0 and float(daily["net_nav"].std(ddof=0)) > 1e-12)
        else 0.0
    )

    turnover = daily['turnover'].mean()
    win_rate = float((metrics_data["net_nav"] > 0).mean())
    profit_sum = float(metrics_data.loc[metrics_data["net_nav"] > 0, "net_nav"].sum())
    loss_sum = float(np.abs(metrics_data.loc[metrics_data["net_nav"] < 0, "net_nav"].sum()))
    profit_ratio = float(profit_sum / loss_sum) if loss_sum > 1e-12 else 0.0
    

    
    ## 月度收益
    month_ret = metrics_data["net_nav"].resample("1M").sum()
    ## 周度收益
    week_ret = metrics_data["net_nav"].resample("1W").sum()
    
    
    return {"ann_ret":ann_ret,"ann_vol":ann_vol,
            "sharpe":sharpe,"calmar":calmar, 
            "win_rate":win_rate, "profit_ratio":profit_ratio,
            "maxdd":maxdd,"turnover":turnover
            }, daily, month_ret, week_ret
    

def plot_result(title_prefix, factor_name, return_name, factor_data, profit_results, profit_daily, profit_month_return, profit_week_return,
                spread_sequence, spread_results, ic_sequence, pred_results, image_path):
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(4, 2, figsize=(16, 24))
        fig.suptitle(title_prefix, fontsize=16)
        
        # Gorss/Net 整体收益
        ax1 = axes[0,0]
        ax1.plot(profit_daily['gross_nav'].index, profit_daily['gross_nav'].cumsum().values, label="Gross NAV", color="orange", linewidth=1.8)
        ax1.plot(profit_daily['net_nav'].index, profit_daily['net_nav'].cumsum().values, label="Net NAV", color="royalblue", linewidth=1.8)
        ax1.set_title("Gross/Net")
        ax1.set_ylabel("NAV")
        ax1.legend(loc="best")
        
        # 指标计算
        ax2 = axes[0, 1]
        ax2.axis("off")
        kpi_fontsize = 12
        # metrics_text = (
        #     f"{'Ann Return':<10}: {(profit_results.get('ann_ret', float('nan')) * 100):.2f}%   "
        #     f"{'Ann Sharpe':<10}: {(profit_results.get('sharpe', float('nan'))):.2f}\n"
        #     f"{'Calmar Ratio':<10}: {(profit_results.get('calmar', float('nan'))):.2f}   "
        #     f"{'Win Rate':<10}: {(profit_results.get('win_rate', float('nan'))):.2f}\n"
        #     f"{'Profit/Loss':<10}: {(profit_results.get('profit_ratio', float('nan'))):.2f}   "
        #     f"{'Max DD':<10}: {(profit_results.get('maxdd', float('nan'))):.2f}\n"
        #     f"{'Turnover':<10}: {(profit_results.get('turnover', float('nan'))):.2f}   "
        # )
        # prediction_text = (
        #     f"{'Total P IC':<10}: {(pred_results.get('total_person_ic', float('nan'))):.2f}   "
        #     f"{'Total R IC':<10}: {(pred_results.get('total_rank_ic', float('nan'))):.2f}\n"
        #     f"{'Mean P IC':<10}: {(pred_results.get('person_ic_mean', float('nan'))):.2f}   "
        #     f"{'Mean R IC':<10}: {(pred_results.get('rank_ic_mean', float('nan'))):.2f}\n"
        #     f"{'P ICIR':<10}: {(pred_results.get('person_icir', float('nan'))):.2f}   "
        #     f"{'R ICIR':<10}: {(pred_results.get('rank_icir', float('nan'))):.2f}\n"
        #     f"{'AutoCorr':<10}: {(factor_data['net_er_out'].autocorr(lag=1)):.2f} "
        # )
        # ax2.set_title("Key Metrics Indicators")
        
        # # 在同一个 ax2 中分层布局：上方左右文本，下方双轴小图
        # metrics_ax = ax2.inset_axes([0.03, 0.56, 0.30, 0.38])
        # prediction_ax = ax2.inset_axes([0.36, 0.56, 0.30, 0.38])
        # metrics_ax.axis("off")
        # prediction_ax.axis("off")
        
        
        # metrics_ax.text(
        #     0.0, 1.0, metrics_text,
        #     transform=metrics_ax.transAxes,
        #     fontsize=kpi_fontsize,
        #     verticalalignment='top',
        #     fontfamily='monospace'
        # )
        # prediction_ax.text(
        #     0.0, 1.0, prediction_text,
        #     transform=prediction_ax.transAxes,
        #     fontsize=kpi_fontsize,
        #     verticalalignment='top',
        #     fontfamily='monospace'
        # )
        
        metrics_text = (
            f"--- Metrics ---\n"
            f"{'Ann Return':<12}: {(profit_results.get('ann_ret', float('nan'))):<10.2f}"
            f"{'Ann Sharpe':<12}: {(profit_results.get('sharpe', float('nan'))):<10.2f}\n"
            f"{'Calmar Ratio':<12}: {(profit_results.get('calmar', float('nan'))):<10.2f}"
            f"{'Win Rate':<12}: {(profit_results.get('win_rate', float('nan'))):<10.2f}\n"
            f"{'Profit/Loss':<12}: {(profit_results.get('profit_ratio', float('nan'))):<10.2f}"
            f"{'Rank IC':<12}: {(pred_results.get('total_rank_ic', float('nan'))):<10.2f}\n"
            # f"{'Max DD':<12}: {(profit_results.get('maxdd', float('nan'))):<10.2f}\n"
            f"{'Turnover':<12}: {(profit_results.get('turnover', float('nan'))):<10.2f}"
            f"{'Persion IC':<12}: {(pred_results.get('total_person_ic', float('nan'))):<10.2f}\n"
            # f"{'Rank IC':<12}: {(pred_results.get('total_rank_ic', float('nan'))):<10.2f}"
            # f"{'Mean P IC':<12}: {(pred_results.get('person_ic_mean', float('nan'))):<10.2f}\n"
            # f"{'Mean R IC':<12}: {(pred_results.get('rank_ic_mean', float('nan'))):<10.2f}"
            # f"{'P ICIR':<12}: {(pred_results.get('person_icir', float('nan'))):<10.2f}\n"
            # f"{'R ICIR':<12}: {(pred_results.get('rank_icir', float('nan'))):<10.2f}"
            # f"{'AutoCorr':<12}: {(factor_data['net_er_out'].autocorr(lag=1)):<10.2f}\n"
        )
        # 上方文本信息区：两列并排，避免与散点图重叠
        metrics_ax = ax2.inset_axes([0.03, 0.75, 0.45, 0.28])
        # prediction_ax = ax2.inset_axes([0.52, 0.68, 0.45, 0.28])
        metrics_ax.axis("off")
        # prediction_ax.axis("off")
        metrics_ax.text(
            0.0, 1.0, metrics_text,
            transform=metrics_ax.transAxes,
            fontsize=kpi_fontsize,
            verticalalignment='top',
            fontfamily='monospace',
            linespacing=1.22
        )
        # prediction_ax.text(
        #     0.0, 1.0, prediction_text,
        #     transform=prediction_ax.transAxes,
        #     fontsize=kpi_fontsize,
        #     verticalalignment='top',
        #     fontfamily='monospace',
        #     linespacing=1.22
        # )
        
        # 下方子图：Scatter Plot（下移，避免标题和文本重叠）
        scatter_ax = ax2.inset_axes([0.03, 0.05, 0.94, 0.60]) # [left, bottom, width, height]
        scat_df = factor_data[[factor_name, return_name]].dropna()
        if len(scat_df) > 0:
            max_points = 10000
            if len(scat_df) > max_points:
                scat_df = scat_df.sample(max_points, random_state=42)
        scatter_x = scat_df[factor_name].values
        scatter_y = scat_df[return_name].values
        scatter_ax.scatter(scatter_x, scatter_y, s=6, alpha=0.18, c="purple", edgecolors="none")
        scatter_ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        scatter_ax.set_title("Factor vs. Return Scatter Plot", pad=2)
        scatter_ax.set_xlabel(factor_name)
        scatter_ax.set_ylabel(return_name)
        scatter_ax.margins(y=0.18) # y轴“长度/拉伸感”
        if len(scatter_y) > 0:
            # 用分位数控制 y 轴，减少极端点影响，同时整体拉长显示区间
            y_min = float(np.nanpercentile(scatter_y, 0.5))
            y_max = float(np.nanpercentile(scatter_y, 99.5))
            y_span = y_max - y_min
            y_pad = max(1e-6, 0.28 * y_span)
            scatter_ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        
        #IC
        ax3 = axes[1,0]
        ic_plot = ic_sequence[['p_ic', 's_ic']].copy().dropna(how='all')
        p_vals = ic_plot['p_ic'].fillna(0.0)
        s_vals = ic_plot['s_ic'].fillna(0.0)
        # 值大的先画，值小的后画，避免小值被完全遮住
        p_first = (np.abs(p_vals.values) >= np.abs(s_vals.values))
        s_first = ~p_first
        ax3.bar(
            ic_plot.index[p_first], p_vals.values[p_first],
            color="steelblue", alpha=0.65, width=1.0, label="Rolling P IC", zorder=1
        )
        ax3.bar(
            ic_plot.index[p_first], s_vals.values[p_first],
            color="seagreen", alpha=0.65, width=1.0, label="Rolling R IC", zorder=2
        )
        ax3.bar(
            ic_plot.index[s_first], s_vals.values[s_first],
            color="seagreen", alpha=0.65, width=1.0, label="_nolegend_", zorder=1
        )
        ax3.bar(
            ic_plot.index[s_first], p_vals.values[s_first],
            color="steelblue", alpha=0.65, width=1.0, label="_nolegend_", zorder=2
        )
        
        ax3.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax3.set_ylabel("Rolling IC")
        ax3.set_title("IC Analysis")
        ax3b = ax3.twinx()
        ax3b.plot(ic_plot.index, p_vals.cumsum().values, color="purple", linewidth=1.8, label="Cumulative P IC")
        ax3b.plot(ic_plot.index, s_vals.cumsum().values, color="red", linewidth=1.8, label="Cumulative R IC")
        ax3b.set_ylabel("Cumulative IC")
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc="best")
        
        
        # 回撤
        ax4 = axes[1, 1]
        dd_plot = profit_daily['maxdd'].astype(float)
     
        ax4.plot(profit_daily['maxdd'].index, dd_plot.values, color="firebrick", linewidth=1.6, label="Drawdown")
        ax4.fill_between(profit_daily['maxdd'].index, dd_plot.values, 0.0, color="firebrick", alpha=0.20)
        ax4.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax4.set_title("Drawdown (from Net NAV)")
        ax4.set_ylabel("Drawdown")
        ax4.legend(loc="best")
        
        
        
        ax5 = axes[2, 0]
        wr_ax = ax5
        pr_ax = wr_ax.twinx()
        win_rate_plot = profit_daily['win_rate'].dropna()
        profit_ratio_plot = profit_daily['profit_ratio'].dropna()
        if len(win_rate_plot) > 0:
            wr_ax.plot(win_rate_plot.index, win_rate_plot.values, color="darkred", linewidth=1.3, label="Win Rate")
        if len(profit_ratio_plot) > 0:
            pr_ax.plot(profit_ratio_plot.index, profit_ratio_plot.values, color="navy", linewidth=1.3, label="Profit/Loss Ratio")
        wr_ax.set_title("Win Rate + Profit/Loss Ratio", fontsize=10)
        wr_ax.set_ylabel("Win Rate", color="darkred", fontsize=9)
        pr_ax.set_ylabel("P/L Ratio", color="navy", fontsize=9)
        wr_ax.tick_params(axis="y", labelcolor="darkred", labelsize=8)
        pr_ax.tick_params(axis="y", labelcolor="navy", labelsize=8)
        wr_ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        wr_ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        wr_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        lines1, labels1 = wr_ax.get_legend_handles_labels()
        lines2, labels2 = pr_ax.get_legend_handles_labels()
        wr_ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
        
        
        
        # Month Week
        ax6 = axes[2,1]
        month_return = profit_month_return.copy()
        month_return.index = pd.to_datetime(month_return.index, errors="coerce")
        month_return = month_return[~month_return.index.isna()].sort_index()
    
        week_return = profit_week_return.copy()
        week_return.index = pd.to_datetime(week_return.index, errors="coerce")
        week_return = week_return[~week_return.index.isna()].sort_index()
        ax6.bar(
            month_return.index,
            month_return.values,
            color=np.where(month_return.values >= 0, "#1F77B4", "#D62728"),
            width=20,
            alpha=0.28,
            label="Monthly Return (Bar)",
            )
        
        line_colors = np.where(week_return.values >= 0, "#2E8B57", "#C0392B")
        ax6.plot(week_return.index, week_return.values, color="#1f77b4", linewidth=1.8, alpha=0.9, label="Weekly Return (Line)")
        ax6.scatter(week_return.index, week_return.values, c=line_colors, s=18, alpha=0.9, zorder=3)
        ax6.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax6.set_title("Weekly (Line) + Monthly (Bar) Return")
        ax6.set_ylabel("Return")
        ax6.legend(loc="best")
        ax6.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        
        
        # 散点图
        # ax4 = axes[1,1]
        # scat_df = factor_data[[factor_name, return_name]].dropna()
        # if len(scat_df) > 0:
        #     max_points = 50000
        #     if len(scat_df) > max_points:
        #         scat_df = scat_df.sample(max_points, random_state=42)
        # scatter_x = scat_df[factor_name].values
        # scatter_y = scat_df[return_name].values
        # ax4.scatter(scatter_x, scatter_y, s=6, alpha=0.18, c="purple", edgecolors="none")
        # ax4.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        # ax4.set_title("Factor vs. Return Scatter Plot")
        # ax4.set_xlabel(factor_name)
        # ax4.set_ylabel(return_name)
        
        
        # Spead
        ax7 = axes[3,0]
        ax7.bar(spread_sequence.index, spread_sequence.values, color="steelblue", alpha=0.65, width=1.0, label="Rolling Spread")
        ax7.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax7.set_ylabel("Rolling Spread")
        ax7.set_title("Spread Analysis")
        ax7b = ax7.twinx()
        ax7b.plot(spread_sequence.index, spread_sequence.cumsum().values, color="purple", linewidth=1.8, label="Cumulative Spread")
        ax7b.set_ylabel("Cumulative Spread")
        lines1, labels1 = ax7.get_legend_handles_labels()
        lines2, labels2 = ax7b.get_legend_handles_labels()
        ax7.legend(lines1 + lines2, labels1 + labels2, loc="best")
        
        
        #IC
        # ax3 = axes[2,0]
        # ax3.bar(ic_sequence['s_ic'].index, ic_sequence['s_ic'].values, color="steelblue", alpha=0.65, width=1.0, label="Rolling IC")
        # ax3.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        # ax3.set_ylabel("Rolling Rank IC")
        # ax3.set_title("IC Analysis")
        # ax3b = ax3.twinx()
        # ax3b.plot(ic_sequence['s_ic'].index, ic_sequence['s_ic'].cumsum().values, color="purple", linewidth=1.8, label="Cumulative IC")
        # ax3b.set_ylabel("Cumulative Rank IC")
        # lines1, labels1 = ax3.get_legend_handles_labels()
        # lines2, labels2 = ax3b.get_legend_handles_labels()
        # ax3.legend(lines1 + lines2, labels1 + labels2, loc="best")
        
        
        
        # ## 胜率
        # ax7 = axes[3, 0]
        # win_rate_plot = profit_daily['win_rate'].dropna()
        # ax7.plot(win_rate_plot.index, win_rate_plot.values, color="darkred", linewidth=1.4)
        # ax7.set_title("Win Rate")
        # ax7.set_ylabel("Rate")
        
        # ## 胜率
        # ax8 = axes[3, 1]
        # profit_ratio_plot = profit_daily['profit_ratio'].dropna()
        # ax8.plot(profit_ratio_plot.index, profit_ratio_plot.values, color="darkred", linewidth=1.4)
        # ax8.set_title("Profit/Loss Ratio")
        # ax8.set_ylabel("Ratio")
        
        
        
        # 换手率
        ax8 = axes[3, 1]
        turnover_plot = profit_daily['turnover'].dropna()
        ax8.plot(turnover_plot.index, turnover_plot.values, color="darkgreen", linewidth=1.4)
        ax8.set_title("Turnover")
        ax8.set_ylabel("Turnover")
        
        
        
        
        for ax in [ax1, ax3, ax5, ax7, ax8]:
            if ax.has_data():
                ax.tick_params(axis="x", rotation=30)
            
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        
        fig.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def create_evaluate(df: pd.DataFrame, factor_name:str,
                    return_name:str = "future_ret_h",
                    pnl_return_name:str = "current_ret",
                    holding_period:int = 15,
                    pnl_method:str = "raw",
                    title_prefix: str = "",
                    cost_rate:float=0.000023,
                    image_path:str=None):
    pnl_method = str(pnl_method).strip().lower()
    pnl_ret_col = return_name if pnl_method == "points_norm" else pnl_return_name

    profit_results, profit_daily, profit_month_return, profit_week_return = profitability(
        data=df[['trade_time', factor_name, pnl_ret_col]],
        factor_name=factor_name,
        return_name=pnl_ret_col,
        cost_rate=cost_rate,
        max_pos=0,
        holding_period=holding_period,
        pnl_method=pnl_method,
    )
    spread_sequence, spread_results = quantile(
        data=df[['trade_time', factor_name, return_name]],
        factor_name=factor_name,
        return_name=return_name,
    )
    ic_sequence, pred_results = pred_metrics(
        data=df[['trade_time', factor_name, return_name]],
        factor_name=factor_name,
        return_name=return_name,
    )
    
    plot_result(title_prefix=title_prefix, 
                factor_name=factor_name,
                return_name=return_name,
                factor_data=df[['trade_time', factor_name, return_name]], profit_results=profit_results, 
                profit_daily=profit_daily, 
                profit_month_return=profit_month_return, 
                profit_week_return=profit_week_return,
                spread_sequence=spread_sequence, 
                spread_results=spread_results, 
                ic_sequence=ic_sequence, 
                pred_results=pred_results,
                image_path=image_path)
    
