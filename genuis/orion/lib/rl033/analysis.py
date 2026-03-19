import pandas as pd
import numpy as np
import pdb,os
from dataclasses import dataclass,field
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lib.cms003.booster import Booster


@dataclass
class TopBottomParams:
    top_k: int
    bottom_k: int
    mode: int
    method: str
    hold: int
    name: str = field(init=False) 

    def __post_init__(self):
        # 此时前面五个参数已经被 dataclass 自动赋值给了 self
        # 可以直接基于它们计算并赋值给 self.name
        self.name = "{0}|{1}m|{2}w|{3}h".format(
            self.top_k,
            self.mode, self.method[0:2],
            self.hold
        )


def normalize_strength(strength: np.ndarray):
    s = np.asarray(strength, dtype=np.float64)
    s = np.where(np.isfinite(s), s, 0.0)
    s = np.clip(s, 0.0, None)
    total = float(np.sum(s))
    if total <= 1e-12:
        return np.ones_like(s, dtype=np.float64) / len(s)
    return s / total


## 单方向权重归一化
def norm_leg(weight: np.ndarray):
    sums = np.nansum(weight, axis=1, keepdims=True)
    out = np.divide(
        weight,
        sums,
        out=np.full_like(weight, np.nan, dtype=np.float64),
        where=(sums > 0),
    )
    return out

## 分数转权重
def scores_to_weights(values: np.ndarray, side: str, weight_method: str):
    if weight_method == "equal":
        return np.ones(len(values), dtype=np.float64) / len(values)
    
    if side == "long":
        ref = values
    elif side == "short":
        ref = -values
    
    if weight_method == "linear":
        strength = ref - np.min(ref)
        return normalize_strength(strength + 1e-12)
    elif weight_method == "sqrt":
        strength = np.sqrt(np.clip(ref - np.min(ref), 0.0, None) + 1e-12)
        return normalize_strength(strength)
    
    raise ValueError("weight_method must be one of: equal/linear/sqrt")
    

def top_bottom_weight(booster,  factors, top_k: int, bottom_k: int, mode:int,
                      method:str, hold:int):

    t_size, n_size = factors.shape
    long_w = np.full((t_size, n_size), np.nan, dtype=np.float64)
    short_w = np.full((t_size, n_size), np.nan, dtype=np.float64)
    
    if mode == 0:
        raw_full = factors.copy()
    elif mode == 1:
        raw_full = booster.score(factors.copy(), None, "ppf")
    
    for t in range(t_size):
        row_for_pick = raw_full[t]
        valid_idx = np.where(~np.isnan(row_for_pick))[0]
        n_valid = len(valid_idx)
        if n_valid == 0:
            continue
        tk = min(int(top_k), n_valid)
        bk = min(int(bottom_k), max(0, n_valid - tk))
        if tk <= 0 or bk <= 0:
            continue
        order = valid_idx[np.argsort(row_for_pick[valid_idx])]  # 小->大
        bot_idx = order[:bk]
        top_idx = order[-tk:]
        
        if mode == 0:
            top_vals = raw_full[t, top_idx]
            bot_vals = raw_full[t, bot_idx]
        elif mode == 1:
            ## 重新打分
            row_top = np.full((1, n_size), np.nan, dtype=np.float64)
            row_top[0, top_idx] = raw_full[t, top_idx]
            row_bot = np.full((1, n_size), np.nan, dtype=np.float64)
            row_bot[0, bot_idx] = raw_full[t, bot_idx]
            top_vals = booster.score(row_top, None, "ppf")[0, top_idx]
            bot_vals = booster.score(row_bot, None, "ppf")[0, bot_idx]
            
        long_w[t, top_idx] = scores_to_weights(
            top_vals, side="long", weight_method=method
        )
        
        short_w[t, bot_idx] = scores_to_weights(
            bot_vals, side="short", weight_method=method
        )
        
    if hold > 1:
        long_w = booster.smooth_hold(weight=long_w, hold=hold)
        short_w = booster.smooth_hold(weight=short_w, hold=hold)
        long_w = norm_leg(long_w)
        short_w = norm_leg(short_w)

    both_w = np.where(
        np.isnan(long_w) & np.isnan(short_w),
        np.nan,
        np.nan_to_num(long_w, nan=0.0) - np.nan_to_num(short_w, nan=0.0)
    )
    
    return long_w, short_w, both_w
        

def evaluate_topbottom(df:pd.DataFrame, factor_name: str, return_name: str, 
                       cost_rate:float, 
                       params, # top_k, bottom_k,order_mode,weight_method,hold
                       skip: int = 0,
                       annual_days: int=365):
    booster = Booster(0, 0, 0, 0) ## 构造函数都未生效
    work = df.copy().set_index(['trade_time','code'])
    work = work.unstack()
    factors = work[factor_name]
    returns = work[return_name]
    
    ts_index = pd.to_datetime(factors.index)
    ereturns = booster.yields(returns.values.copy(), None, skip, -1) # 绝对收益
    # columns = factors.columns
    res = []
    for param in params:
        long_weight, short_weight, both_weight = top_bottom_weight(
            booster=booster,factors=factors.values, 
            top_k=10, bottom_k=10,
            mode=param.mode,
            method=param.method, 
            hold=param.hold)
        ## 计算
        w0 = np.nan_to_num(both_weight, nan=0.0)
        period_gross_nav = np.nansum(both_weight * ereturns, axis=1)
        period_turnover = np.sum(np.abs(w0[1:] - w0[:-1]), axis=1) * 0.5
        period_turnover = np.concatenate([[0.0], period_turnover])
    
        ## 每期毛收益率
        period_gross_nav = np.nansum(both_weight * ereturns, axis=1)
    
        ## 每期换手率
        period_turnover = np.sum(np.abs(w0[1:] - w0[:-1]), axis=1) * 0.5
        period_turnover = np.concatenate([[0.0], period_turnover])
    
        ## 净收益
        period_net_nav = period_gross_nav - cost_rate * period_turnover
        
        ## 净值
        nav = np.cumprod(1.0 + period_net_nav)
        metrics_data = pd.DataFrame(
            {'turnover':period_turnover,
            'gross_nav':period_gross_nav,
            'net_nav':period_net_nav,
            'nav':nav},index=ts_index)
    
        # ## 月度收益
        # month_ret = metrics_data["net_nav"].resample("1M").sum()
        # ## 周度收益
        # week_ret = metrics_data["net_nav"].resample("1W").sum()
    
        daily = pd.DataFrame(index=metrics_data.resample("1D").size().index)
        daily.index.name = "trade_date"
        daily["turnover"] = metrics_data["turnover"].resample("1D").mean()
        daily["gross_nav"] = metrics_data["gross_nav"].resample("1D").sum()
        daily["net_nav"] = metrics_data["net_nav"].resample("1D").sum()
        daily["nav"] = np.exp(daily["net_nav"].cumsum())
        daily["pnl"] = np.nancumsum(daily["net_nav"].values)
        daily["maxdd"] = np.maximum.accumulate(daily["pnl"].values) - daily["pnl"].values
    
        ann_ret = daily["net_nav"].mean() * annual_days
        ann_vol = daily["net_nav"].std(ddof=0) * np.sqrt(annual_days)
        maxdd = np.nanmax(daily["maxdd"].values)
        calmar = ann_ret / maxdd
        sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
        turnover = daily['turnover'].mean()
        
        res.append({
            "metrics": { 
                "ann_ret":ann_ret,
                "maxdd":maxdd,
                "calmar":calmar, 
                "sharpe":sharpe, 
                "turnover":turnover,
                "ann_vol":ann_vol
                },
            "name": param.name,
            "net_nav":daily["net_nav"]
        })
        
    return res
    
    
    

def calc_sequence(factors, returns, hold, skip, quantiles, category, cost_rate, method):
    booster = Booster(hold, skip, 0, category)
    score = booster.score(factors=factors.values.copy(), dummy=None, method=method)
    ereturns = booster.yields(returns.values.copy(), None, skip, category)
    # ## 用于分位数计算
    ereturns1 = booster.yields(returns.values.copy(), None, skip, 1)
    
    
    right = booster.create_weight(score, is_pos=True)
    left = booster.create_weight(score, is_pos=False)
    ## 平滑处理
    if hold > 1:
        right = booster.smooth_hold(weight=right, hold=hold)
        left = booster.smooth_hold(weight=left, hold=hold)
    
    long_weight, short_weight, both_weight, _ = booster.direction(
        right_weight=right, left_weight=left, direction=1
    )
    
    ## 换手率
    w0 = np.nan_to_num(both_weight, nan=0.0)
    # turnover = float(np.mean(np.sum(np.abs(w0[1:] - w0[:-1]), axis=1) * 0.5))
    
    # IC / ICIR（only both）
    ic_arr, _, _ = booster.correlation(both_weight, ereturns, 'both')
    # icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
    
    ## 每期毛收益率
    period_gross_nav = np.nansum(both_weight * ereturns, axis=1)
    
    ## 每期换手率
    period_turnover = np.sum(np.abs(w0[1:] - w0[:-1]), axis=1) * 0.5
    period_turnover = np.concatenate([[0.0], period_turnover])
    
    ## 净收益
    period_net_nav = period_gross_nav - cost_rate * period_turnover
    
    ## 净值
    nav = np.cumprod(1.0 + period_net_nav)
    
    
    ## 分位
    quantile_serise1 = {}
    quantile_serise2 = {}
    pct_ranks = booster.percent_rank(score)
    for q in range(1, quantiles + 1):
        lower_bound = (q - 1) / quantiles
        upper_bound = q / quantiles
        mask = (pct_ranks > lower_bound) & (pct_ranks <= upper_bound)
        qw = np.where(mask, 1.0, 0.0)
        
        row_sums = np.sum(qw, axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            qw = np.where(row_sums > 0, qw / row_sums, 0.0)
        if hold > 1:
            qw = pd.DataFrame(qw).rolling(
                hold, min_periods=1).sum().div(hold).values
            row_sums_smooth = np.nansum(qw, axis=1, keepdims=True)
            qw = np.where(row_sums_smooth > 0, qw / row_sums_smooth,
                                  0.0)
        
        ## 分组收益序列
        q_rets1 = np.nansum(ereturns * qw, axis=1)
        q_rets2 = np.nansum(ereturns1 * qw, axis=1)
        quantile_serise1[f"Q{q}"] = pd.Series(q_rets1, index=returns.index)
        quantile_serise2[f"Q{q}"] = pd.Series(q_rets2, index=returns.index)
    
    quantile_serise1["Q"] = quantile_serise1[f"Q{quantiles}"] - quantile_serise1[f"Q1"]
    quantile_serise2["Q"] = quantile_serise2[f"Q{quantiles}"] - quantile_serise2[f"Q1"]
        
    
    ts_index = factors.index  
    columns = factors.columns
    
    metrics_data = pd.DataFrame(
        {'ic':ic_arr, 
         'turnover':period_turnover,
         'gross_nav':period_gross_nav,
         'net_nav':period_net_nav,
         'nav':nav},index=ts_index)
    
    long_weight_data = pd.DataFrame(long_weight, index=ts_index, columns=columns)
    short_weight_data = pd.DataFrame(short_weight, index=ts_index, columns=columns)
    both_weight_data = pd.DataFrame(both_weight, index=ts_index, columns=columns)
    
    weights_data = pd.concat(
        {"long": long_weight_data, "short": short_weight_data, "both": both_weight_data},
        axis=1)
    weights_data = weights_data.stack().reset_index()
    
    return metrics_data, weights_data, quantile_serise1, quantile_serise2
    

def daily_metrics(metrics_data, annual_days=365):
    ## 转化日频
    metrics_data = metrics_data.copy()
    metrics_data.index = pd.to_datetime(metrics_data.index)
    metrics_data = metrics_data.sort_index()
    
    daily = pd.DataFrame(index=metrics_data.resample("1D").size().index)
    daily.index.name = "trade_date"
    
    daily["ic"] = metrics_data["ic"].resample("1D").mean()
    daily["turnover"] = metrics_data["turnover"].resample("1D").mean()
    
    daily["gross_nav"] = metrics_data["gross_nav"].resample("1D").sum()
    
    daily["net_nav"] = metrics_data["net_nav"].resample("1D").sum()
    
    daily["nav"] = np.exp(daily["net_nav"].cumsum())
    
    daily["pnl"] = np.nancumsum(daily["net_nav"].values)
    
    daily["maxdd"] = np.maximum.accumulate(daily["pnl"].values) - daily["pnl"].values
    
    ## 预测性评估
    ic_mean = daily['ic'].mean()
    ic_std = daily['ic'].std()
    icir = ic_mean / ic_std
    
    ## 绩效评估
    ann_ret = daily["net_nav"].mean() * annual_days
    ann_vol = daily["net_nav"].std(ddof=0) * np.sqrt(annual_days)
    maxdd = np.nanmax(daily["maxdd"].values)
    calmar = ann_ret / maxdd
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    turnover = daily['turnover'].mean()

    
    ## 月度收益
    month_ret = metrics_data["net_nav"].resample("1M").sum()
    ## 周度收益
    week_ret = metrics_data["net_nav"].resample("1W").sum()
    return {"ic_mean":ic_mean, "icir":icir, "ann_ret":ann_ret,
             "maxdd":maxdd,"calmar":calmar, 
            "sharpe":sharpe, "turnover":
                turnover,"ann_vol":ann_vol}, daily, month_ret, week_ret
      
    
def evaluate_portfolio(df:pd.DataFrame, factor_name: str,
                       return_name: str, 
                       method:str='ppf', 
                       hold:int = 1, 
                       skip: int = 0,
                       quantiles:int = 20,
                       cost_rate=0.0002,
                       category: int = -1):
    
    work = df.copy().set_index(['trade_time','code'])
    work = work.unstack()
    factors = work[factor_name]
    returns = work[return_name]
    
    
    metrics_data, weights_data, quantile_serise1, quantile_serise2 = calc_sequence(
        factors=factors, returns=returns, 
        hold=hold, skip=skip, quantiles=quantiles, 
        category=category,  cost_rate=cost_rate, 
        method=method)
    
    result, daily, month_ret, week_ret = daily_metrics(metrics_data)
    factor_data = df[['trade_time','score','nxt1_ret']]
    return result, daily, quantile_serise1, quantile_serise2, factor_data, month_ret, week_ret
   
  
    
def plot_results(title_prefix, quantiles, result_metrics, 
                 daily_metrics, 
                 quantile_serise1, quantile_serise2,
                 factor_data, month_return, 
                 week_return, top_bottom_res):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(5, 2, figsize=(32, 24))
    fig.suptitle(title_prefix, fontsize=16)
    
    # Gorss/Net 整体收益
    ax1 = axes[0, 0]
    ax1.plot(daily_metrics['gross_nav'].index, daily_metrics['gross_nav'].cumsum().values, label="Gross NAV", color="orange", linewidth=1.8)
    ax1.plot(daily_metrics['net_nav'].index, daily_metrics['net_nav'].cumsum().values, label="Net NAV", color="royalblue", linewidth=1.8)
    ax1.set_title("Metrics")
    ax1.set_ylabel("NAV")
    ax1.legend(loc="best")
    
    
    # 指标计算
    ax2 = axes[0, 1]
    ax2.axis("off")
    kpi_fontsize = 12

    def _fmt(v):
        if v is None:
            return "N/A"
        try:
            if np.isnan(v):
                return "N/A"
        except Exception:
            pass
        return f"{float(v):.4f}"

    metrics_order = [
        ("Ann Return", "ann_ret"),
        ("Max Drawdown", "maxdd"),
        ("Calmar", "calmar"),
        ("Sharpe", "sharpe"),
        ("IC Mean", "ic_mean"),
        ("ICIR", "icir"),
        ("Turnover", "turnover"),
        ("Ann Vol", "ann_vol"),
    ]

    # 列数据：整体 + 各 TopBottom
    columns = [("Overall", result_metrics)]
    if top_bottom_res:
        for tb in top_bottom_res:
            columns.append((tb.get("name", "TopBottom"), tb.get("metrics", {})))

    metric_w = 13
    col_w = 12
    header = f"{'Metric':<{metric_w}} | " + " | ".join([f"{name[:col_w]:<{col_w}}" for name, _ in columns])
    sep = "-" * len(header)
    lines = [header, sep]

    for label, key in metrics_order:
        row_vals = []
        for col_name, col_data in columns:
            if col_name != "Overall" and key in ("ic_mean", "icir"):
                row_vals.append(f"{'N/A':<{col_w}}")
            else:
                row_vals.append(f"{_fmt(col_data.get(key)):<{col_w}}")
        lines.append(f"{label:<{metric_w}} | " + " | ".join(row_vals))

    text2 = "\n".join(lines)
    ax2.text(
        0.02,
        0.95,
        text2,
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=kpi_fontsize,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    ax2.set_title("Key Metrics Indicators")
    
    # IC
    ax3 = axes[1,0]
    ax3.bar(daily_metrics['ic'].index, daily_metrics['ic'].values, color="steelblue", alpha=0.65, width=1.0, label="Rolling IC")
    ax3.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax3.set_ylabel("Rolling IC")
    ax3.set_title("IC Analysis")
    ax3b = ax3.twinx()
    ax3b.plot(daily_metrics['ic'].index, daily_metrics['ic'].cumsum().values, color="purple", linewidth=1.8, label="Cumulative IC")
    ax3b.set_ylabel("Cumulative IC")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="best")
    
    
    # 相对分位图
    ax4 = axes[1, 1]
    cmap = plt.get_cmap("coolwarm_r")
    q_keys = list(quantile_serise2.keys())
    colors = cmap(np.linspace(0, 1, quantiles+1))
    for i, key in enumerate(q_keys):
        nav = quantile_serise2[key].copy()
        nav.index = pd.to_datetime(nav.index, errors="coerce")
        nav = nav[~nav.index.isna()].sort_index()
        ax4.plot(nav.index, nav.cumsum().values, color=colors[i], alpha=0.85, linewidth=1.5, label=key)
    
    ax4.set_title(f"Quantile Cumulative(Relatively) Returns ({quantiles} groups)")
    ax4.set_ylabel("NAV")
    ax4.legend(loc="best", ncol=min(quantiles + 1, 6))
    ax4.grid(True, alpha=0.35)
    ax4.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    
    # top bottom 各收益
    ax5 = axes[2,0]
    colors = cmap(np.linspace(0, 1, len(top_bottom_res)+1))
    for i, tb in  enumerate(top_bottom_res):
        ax5.plot(tb['net_nav'].index, tb['net_nav'].cumsum().values, label="{0} NAV".format(tb['name']), color=colors[i], linewidth=1.8)
    ax5.set_title("TopBottom")
    ax5.set_ylabel("NAV")
    ax5.legend(loc="best")
    
    # 绝对分位图
    ax6 = axes[2, 1]
    cmap = plt.get_cmap("coolwarm_r")
    q_keys = list(quantile_serise1.keys())
    colors = cmap(np.linspace(0, 1, quantiles+1))
    for i, key in enumerate(q_keys):
        nav = quantile_serise1[key].copy()
        nav.index = pd.to_datetime(nav.index, errors="coerce")
        nav = nav[~nav.index.isna()].sort_index()
        ax6.plot(nav.index, nav.cumsum().values, color=colors[i], alpha=0.85, linewidth=1.5, label=key)
    ax6.set_title(f"Quantile Cumulative(Absolute) Returns ({quantiles} groups)")
    ax6.set_ylabel("NAV")
    ax6.legend(loc="best", ncol=min(quantiles + 1, 6))
    ax6.grid(True, alpha=0.35)
    ax6.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    
    ## 回撤
    ax7 = axes[3, 0]
    dd_plot = np.exp(-daily_metrics['maxdd']) - 1.0
    ax7.plot(daily_metrics['maxdd'].index, dd_plot.values, color="firebrick", linewidth=1.6, label="Drawdown")
    ax7.fill_between(daily_metrics['maxdd'].index, dd_plot.values, 0.0, color="firebrick", alpha=0.20)
    ax7.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax7.set_title("Drawdown (from Net NAV)")
    ax7.set_ylabel("Drawdown")
    ax7.legend(loc="best")
    
    
    ## 月度收益
    ## 周度收益
    ax8 = axes[3, 1]
    month_return = month_return.copy()
    month_return.index = pd.to_datetime(month_return.index, errors="coerce")
    month_return = month_return[~month_return.index.isna()].sort_index()
    week_return = week_return.copy()
    week_return.index = pd.to_datetime(week_return.index, errors="coerce")
    week_return = week_return[~week_return.index.isna()].sort_index()
    ax8.bar(
        month_return.index,
        month_return.values,
        color=np.where(month_return.values >= 0, "#1F77B4", "#D62728"),
        width=20,
        alpha=0.28,
        label="Monthly Return (Bar)",
    )
    line_colors = np.where(week_return.values >= 0, "#2E8B57", "#C0392B")
    ax8.plot(week_return.index, week_return.values, color="#1f77b4", linewidth=1.8, alpha=0.9, label="Weekly Return (Line)")
    ax8.scatter(week_return.index, week_return.values, c=line_colors, s=18, alpha=0.9, zorder=3)
    ax8.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax8.set_title("Weekly (Line) + Monthly (Bar) Return")
    ax8.set_ylabel("Return")
    ax8.legend(loc="best")
    ax8.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax8.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
    
    # 换手率
    ax9 = axes[4, 0]
    ax9.plot(daily_metrics['turnover'].index, daily_metrics['turnover'].values, color="darkgreen", linewidth=1.4)
    ax9.set_title("Turnover")
    ax9.set_ylabel("Turnover")
    
    
    
    ax10 = axes[4, 1]
    # 散点图
    scat_df = factor_data[['score', 'nxt1_ret']].dropna()
    if len(scat_df) > 0:
        max_points = 30000
        if len(scat_df) > max_points:
            scat_df = scat_df.sample(max_points, random_state=42)
        scatter_x = scat_df['score'].values
        scatter_y = scat_df['nxt1_ret'].values
    ax10.scatter(scatter_x, scatter_y, s=6, alpha=0.18, c="purple", edgecolors="none")
    ax10.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax10.set_title("Factor vs. Return Scatter Plot")
    ax10.set_xlabel('score')
    ax10.set_ylabel('nxt1_ret')
    
    

    
    for ax in [ax1, ax3, ax4, ax6, ax7, ax8]:
        if ax.has_data():
            ax.tick_params(axis="x", rotation=30)
            
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    
    
    image_path = os.path.join('./', "evaluation_plot.png")
    fig.savefig(image_path, dpi=300)
    # if image_export_dir is not None:
    #     os.makedirs(image_export_dir, exist_ok=True)
    #     export_path = os.path.join(image_export_dir, "evaluation_plot.png")
    #     fig.savefig(export_path, dpi=300)
    plt.close(fig)
    

def create_evaluate(df:pd.DataFrame, factor_name: str,
                       return_name: str, 
                       top_bottom_params: list,
                       method:str='ppf', 
                       hold:int = 1, 
                       skip: int = 0,
                       quantiles:int = 20,
                       cost_rate=0.0002,
                       category: int = -1):
    ## quantile_serise1 绝对收益分位
    ## quantile_serise2 相对收益分位
    result,daily,quantile_serise1,quantile_serise2, factor_data,month_ret,week_ret = evaluate_portfolio(
        df=df, factor_name=factor_name,
        return_name=return_name,
        method=method,
        hold=hold,
        skip=skip,
        quantiles=quantiles,
        cost_rate=cost_rate,
        category=category
    )
    pdb.set_trace()
    top_bottom_res = evaluate_topbottom(df=df, factor_name=factor_name, return_name=return_name,
                       cost_rate=cost_rate,params=top_bottom_params)
    
    plot_results(title_prefix="test", quantiles=quantiles, result_metrics=result, 
                 daily_metrics=daily, 
                 quantile_serise1=quantile_serise1, quantile_serise2=quantile_serise2,
                 factor_data=factor_data, month_return=month_ret, 
                 week_return=week_ret, top_bottom_res=top_bottom_res)
    
    