import numpy as np
from ultron.tradingday import *
from kdutils.data import *

INDICATOR_META = {
    "score": {
        "meaning": "综合评分，汇总波动空间、成本覆盖、趋势质量、流动性和跳跃惩罚。",
        "role": "用于每日横截面排序和生成候选品种池。",
        "monotonicity": "higher_better",
    },
    "rank": {
        "meaning": "每日按 score 从高到低的横截面排名。",
        "role": "用于选取 TopN 或 TopPct 品种。",
        "monotonicity": "lower_better",
    },
    "is_high_vol": {
        "meaning": "最终是否入选高波动/高可交易性品种池。",
        "role": "作为下游策略是否允许交易该品种的布尔过滤器。",
        "monotonicity": "true_better",
    },
    "rows": {
        "meaning": "滚动窗口内有效分钟 K 数量。",
        "role": "衡量样本充分性，避免数据太少导致指标不稳定。",
        "monotonicity": "higher_better_until_sufficient",
    },
    "active_days": {
        "meaning": "滚动窗口内有有效分钟数据的交易日数量。",
        "role": "衡量交易日覆盖度，避免只靠少数几天样本参与排名。",
        "monotonicity": "higher_better_until_sufficient",
    },
    "first_close": {
        "meaning": "滚动窗口内第一个有效收盘价。",
        "role": "用于计算窗口净位移和趋势效率。",
        "monotonicity": "neutral",
    },
    "last_close": {
        "meaning": "滚动窗口内最后一个有效收盘价。",
        "role": "用于计算窗口净位移和趋势效率。",
        "monotonicity": "neutral",
    },
    "avg_amp_pct": {
        "meaning": "平均分钟振幅，等于 mean((high-low)/close)*100。",
        "role": "衡量每根分钟 K 平均能提供多少交易空间。",
        "monotonicity": "higher_better",
    },
    "p80_amp_pct": {
        "meaning": "分钟振幅 80 分位的滚动均值。",
        "role": "衡量较活跃分钟的波动空间，避免均值被大量平淡分钟压低。",
        "monotonicity": "higher_better",
    },
    "atr_pct": {
        "meaning":
        "真实波幅 TR/close 的滚动均值，TR 同时考虑 high-low、high-prev_close、low-prev_close。",
        "role": "衡量包含跳动/跳空在内的有效波动空间。",
        "monotonicity": "higher_better",
    },
    "realized_vol_ann_pct": {
        "meaning": "分钟收益率标准差年化后的实现波动率。",
        "role": "衡量价格变化强度和频率。",
        "monotonicity": "higher_better_with_risk_limit",
    },
    "tsi": {
        "meaning": "趋势信噪比，近似 mean(log_ret)/std(log_ret)。",
        "role": "衡量单位噪声下的平均方向性收益。",
        "monotonicity": "abs_higher_better_directional",
    },
    "cum_tsi": {
        "meaning": "累积趋势信噪比，tsi*sqrt(有效收益样本数)。",
        "role": "体现“以时间换空间”，衡量窗口累积后趋势能否战胜噪声。",
        "monotonicity": "abs_higher_better_directional",
    },
    "rho": {
        "meaning": "短期惯性，近似收益率与滞后收益率的滚动相关/AR(1) 系数。",
        "role": "判断价格更偏趋势延续还是短期反转。",
        "monotonicity": "higher_better_for_trend",
    },
    "trend_quality": {
        "meaning": "趋势可交易性评分，综合 abs(cum_tsi)、正 rho 和 trend_efficiency。",
        "role": "把趋势信噪比、短期惯性、路径效率合成为趋势质量因子。",
        "monotonicity": "higher_better",
    },
    "roundtrip_cost_bp": {
        "meaning": "往返交易成本，单位 bp，包含手续费和统一滑点估计。",
        "role": "衡量交易摩擦，供成本覆盖类指标使用。",
        "monotonicity": "lower_better",
    },
    "amp_to_cost": {
        "meaning": "平均分钟振幅 / 往返成本。",
        "role": "衡量波动空间覆盖交易成本的倍数。",
        "monotonicity": "higher_better",
    },
    "opportunity_after_cost_pct": {
        "meaning": "扣除往返成本后的平均可用振幅空间。",
        "role": "判断看似有波动的品种是否仍有可交易空间。",
        "monotonicity": "higher_better",
    },
    "trend_efficiency": {
        "meaning": "abs(log(last_close/first_close)) / sum(abs(log_ret))。",
        "role": "衡量路径是否顺滑；越高越偏单边趋势，越低越偏来回震荡。",
        "monotonicity": "higher_better_for_trend",
    },
    "jump_share": {
        "meaning": "疑似跳跃波动占比，atr_part > amp*2.5 的样本比例。",
        "role": "惩罚由跳空/断点贡献的波动，避免不可连续交易的假高波动。",
        "monotonicity": "lower_better",
    },
    "max_drawdown": {
        "meaning": "滚动窗口内按日 close 计算的最大回撤，通常为负数。",
        "role": "描述窗口内单边下行风险或回撤压力。",
        "monotonicity": "context_dependent_less_negative_is_safer",
    },
    "median_volume": {
        "meaning": "窗口内成交量中位数。",
        "role": "衡量交易活跃度。",
        "monotonicity": "higher_better_until_liquid",
    },
    "median_value": {
        "meaning": "窗口内成交额中位数。",
        "role": "衡量跨品种更可比的流动性。",
        "monotonicity": "higher_better_until_liquid",
    },
    "median_openint": {
        "meaning": "窗口内持仓量中位数。",
        "role": "衡量合约承载资金和主力稳定性。",
        "monotonicity": "higher_better_until_liquid",
    },
}


def indicator_description() -> pd.DataFrame:
    """返回指标含义、作用和单调性说明。"""
    return pd.DataFrame.from_dict(
        INDICATOR_META,
        orient="index").reset_index().rename(columns={"index": "indicator"})


def _robust_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad < 1e-12:
        std = s.std()
        if not np.isfinite(std) or std < 1e-12:
            return pd.Series(0.0, index=s.index)
        return ((s - s.mean()) / std).fillna(0.0)
    return ((s - med) / mad).fillna(0.0)


def robust_z_by_day(frame: pd.DataFrame) -> pd.DataFrame:
    """对每个交易日横截面做 robust z。"""
    return frame.apply(_robust_z, axis=1)


def interval_trading_date(begin_date: datetime.date, end_date: datetime):
    dates = makeSchedule(begin_date,
                         end_date,
                         '1b',
                         calendar='china.sse',
                         dateRule=BizDayConventions.Following,
                         dateGenerationRule=DateGeneration.Backward)
    return dates


def previous_trading_date(trading_date: datetime.date) -> datetime.date:
    previous_day = advanceDateByCalendar("china.sse", trading_date, "-1b")
    if isinstance(previous_day, datetime.datetime):
        return previous_day.date()
    if isinstance(previous_day, pd.Timestamp):
        return previous_day.date()
    return previous_day


def trade_day_index(
    trade_time_index: pd.Index, overnight_session_end=datetime.time(2, 30)
) -> pd.Index:
    """
    直接从 trade_time 转交易日：
    - 00:00~02:30 属于前一个交易日；
    - 其它时间属于自然日。
    """
    trade_time = pd.Series(pd.to_datetime(trade_time_index),
                           index=trade_time_index)
    trade_day = trade_time.dt.date

    mask = trade_time.dt.time <= overnight_session_end
    if mask.any():
        prev_map = {
            d: previous_trading_date(d)
            for d in trade_day[mask].unique()
        }
        trade_day.loc[mask] = trade_day.loc[mask].map(prev_map)

    return pd.Index(pd.to_datetime(trade_day), name="trade_date")


def fetch_market(codes, begin_time, end_time, adjusted_method):
    # market_data = fetch_main_market(begin_date=start_time,
    #                                 end_date=end_time,
    #                                 codes=codes,
    #                                 method=adjusted_method,
    #                                 keep_symbol=True)

    # end_time = datetime.datetime(2026, 6, 5)
    # start_time = datetime.datetime(2026, 6, 1)
    start_time = advanceDateByCalendar('china.sse', begin_time, '-1b')
    market_data = fetch_main_market(begin_date=start_time,
                                    end_date=end_time,
                                    codes=codes,
                                    method=adjusted_method,
                                    keep_symbol=True,
                                    forced_alignment=True)
    # market_data = fetch_local_market1(base_path=os.environ['BAR_FUT_DIRS'],
    #                                   begin_date=start_time,
    #                                   end_date=end_time,
    #                                   codes=codes,
    #                                   method=adjusted_method,
    #                                   keep_symbol=True)
    market_data = market_data[(market_data['trade_time'] > begin_time)
                              & (market_data['trade_time'] <= end_time)]
    return market_data


def fetch_basic(codes, begin_time, end_time):
    basic_infos = fetch_basic2(
        begin_date=advanceDateByCalendar('china.sse', begin_time, '-60b'),
        end_date=advanceDateByCalendar('china.sse', end_time, '90b'),
        codes=codes,  #[0:5],
        columns=[
            'contractObject', 'code', 'exchangeCD', 'contMultNum',
            'lastTradeDate', 'tradeCommiNum'
        ])
    basic_infos['tradeCommiNum'] = basic_infos['tradeCommiNum'] / 100
    return basic_infos


def calc_indicator(market_matrix,
                   min_rows,
                   default_cost,
                   slippage_bp,
                   window_days,
                   min_periods,
                   min_active_days=None,
                   min_amp_pct=0.1,
                   min_atr_pct=0.1,
                   min_amp_to_cost=1.0,
                   high_vol_top_n=30,
                   high_vol_top_pct=None,
                   use_abs_threshold=False,
                   min_rho=0.0,
                   min_abs_cum_tsi=0.0):
    if 'symbol' in market_matrix.columns.get_level_values(0):
        same_contract = market_matrix['symbol'].eq(
            market_matrix['symbol'].shift(1))
        prev_close = market_matrix['close'].shift(1).where(same_contract)
    else:
        prev_close = market_matrix['close'].shift(1)

    ratio = (market_matrix["close"] / prev_close).where(prev_close > 0)

    amp = (market_matrix["high"] -
           market_matrix["low"]) / market_matrix["close"]

    tr = pd.concat(
        [
            market_matrix["high"] - market_matrix["low"],
            (market_matrix["high"] - prev_close).abs(),
            (market_matrix["low"] - prev_close).abs(),
        ],
        axis=0,
        keys=["hl", "hc", "lc"],
    ).groupby(level=1).max()
    atr_part = tr / market_matrix['close']

    log_ret = np.log(ratio.where(ratio > 0))
    abs_log_ret = log_ret.abs()
    lag_log_ret = log_ret.shift(1)

    # default_cost = basic_infos.groupby(
    #     'code')['tradeCommiNum'].median() + slippage_bp

    roundtrip_cost = market_matrix['tradeCommiNum'].add(slippage_bp)
    roundtrip_cost = roundtrip_cost.fillna(default_cost)

    opportunity = (amp - roundtrip_cost).clip(lower=0.0)
    jump_flag = (atr_part > amp * 2.5).astype(float)

    ## 日频
    trade_day = trade_day_index(market_matrix['close'].index)

    daily_amp_sum = amp.groupby(trade_day).sum()
    daily_amp_p80 = amp.groupby(trade_day).quantile(0.80)
    daily_atr_sum = atr_part.groupby(trade_day).sum()
    daily_ret_sum = log_ret.groupby(trade_day).sum()
    daily_ret_sumsq = log_ret.pow(2).groupby(trade_day).sum()
    daily_ret_count = log_ret.notna().astype("int64").groupby(trade_day).sum()
    daily_lag_sum = lag_log_ret.groupby(trade_day).sum()
    daily_lag_sumsq = lag_log_ret.pow(2).groupby(trade_day).sum()
    daily_ret_lag_prod = log_ret.mul(lag_log_ret).groupby(trade_day).sum()
    daily_pair_count = log_ret.notna().mul(
        lag_log_ret.notna()).astype("int64").groupby(trade_day).sum()
    daily_path_sum = abs_log_ret.groupby(trade_day).sum()
    daily_opp_sum = opportunity.groupby(trade_day).sum()
    daily_jump_sum = jump_flag.groupby(trade_day).sum()

    daily_rows = market_matrix['close'].notna().astype("int64").groupby(
        trade_day).sum()
    rows_60d = daily_rows.rolling(window_days, min_periods=min_periods).sum()

    daily_active = market_matrix['close'].notna().groupby(
        trade_day).any().astype("int64")

    active_days_60d = daily_active.rolling(window_days,
                                           min_periods=min_periods).sum()

    daily_first = market_matrix['close'].groupby(trade_day).first().bfill(
        limit=window_days - 1).shift(window_days - 1)
    daily_last = market_matrix['close'].groupby(trade_day).last().ffill(
        limit=window_days - 1)

    ### 滚动计算
    rolling_peak = daily_last.rolling(window_days,
                                      min_periods=min_periods).max()
    rolling_max_drawdown = (daily_last.div(rolling_peak) - 1.0).rolling(
        window_days, min_periods=min_periods).min()

    rolling_avg_amp_60d = daily_amp_sum.rolling(
        window_days, min_periods=min_periods).sum() / rows_60d

    p80_amp_60d = daily_amp_p80.rolling(window_days, min_periods=1).mean()
    atr_60d = daily_atr_sum.rolling(window_days,
                                    min_periods=min_periods).sum() / rows_60d
    total_path_60d = daily_path_sum.rolling(window_days,
                                            min_periods=min_periods).sum()
    opportunity_60d = daily_opp_sum.rolling(
        window_days, min_periods=min_periods).sum() / rows_60d
    jump_share_60d = daily_jump_sum.rolling(
        window_days, min_periods=min_periods).sum() / rows_60d

    ret_sum_60d = daily_ret_sum.rolling(window_days,
                                        min_periods=min_periods).sum()
    ret_sumsq_60d = daily_ret_sumsq.rolling(window_days,
                                            min_periods=min_periods).sum()
    ret_count_60d = daily_ret_count.rolling(window_days,
                                            min_periods=min_periods).sum()
    ret_var_60d = (ret_sumsq_60d -
                   ret_sum_60d.pow(2).div(ret_count_60d)).div(ret_count_60d -
                                                              1)
    ret_std_60d = np.sqrt(ret_var_60d.clip(lower=0))

    lag_sum_60d = daily_lag_sum.rolling(window_days,
                                        min_periods=min_periods).sum()
    lag_sumsq_60d = daily_lag_sumsq.rolling(window_days,
                                            min_periods=min_periods).sum()
    ret_lag_prod_60d = daily_ret_lag_prod.rolling(
        window_days, min_periods=min_periods).sum()
    pair_count_60d = daily_pair_count.rolling(window_days,
                                              min_periods=min_periods).sum()
    cov_60d = (
        ret_lag_prod_60d -
        ret_sum_60d.mul(lag_sum_60d).div(pair_count_60d)).div(pair_count_60d -
                                                              1)
    ret_var_for_rho = (
        ret_sumsq_60d -
        ret_sum_60d.pow(2).div(pair_count_60d)).div(pair_count_60d - 1)
    lag_var_for_rho = (
        lag_sumsq_60d -
        lag_sum_60d.pow(2).div(pair_count_60d)).div(pair_count_60d - 1)
    rho = cov_60d.div(np.sqrt(ret_var_for_rho * lag_var_for_rho)).clip(-1, 1)

    median_volume_60d = market_matrix['volume'].groupby(
        trade_day).median().rolling(window_days, min_periods=1).median()
    median_value_60d = market_matrix['value'].groupby(
        trade_day).median().rolling(window_days, min_periods=1).median()
    median_openint_60d = market_matrix['openint'].groupby(
        trade_day).median().rolling(window_days, min_periods=1).median()

    roundtrip_cost_daily = roundtrip_cost.groupby(trade_day).median()
    roundtrip_cost_60d = roundtrip_cost_daily.ffill().fillna(default_cost)

    periods_per_year = 252.0 * rows_60d.div(
        active_days_60d.clip(lower=1)).clip(lower=1.0)
    total_move = np.log(
        daily_last.div(daily_first).where(daily_first > 0)).abs()

    avg_amp_pct = rolling_avg_amp_60d * 100.0
    p80_amp_pct = p80_amp_60d * 100.0
    atr_pct = atr_60d * 100.0
    realized_vol_ann_pct = ret_std_60d.fillna(0.0) * np.sqrt(
        periods_per_year) * 100.0
    mean_ret_60d = ret_sum_60d.div(ret_count_60d.replace(0, np.nan))
    tsi = mean_ret_60d.div(ret_std_60d.replace(0, np.nan))
    cum_tsi = tsi.mul(np.sqrt(ret_count_60d))
    roundtrip_cost_bp = roundtrip_cost_60d * 10000.0
    amp_to_cost = rolling_avg_amp_60d.div(roundtrip_cost_60d.replace(
        0, np.nan))
    opportunity_after_cost_pct = opportunity_60d * 100.0
    trend_efficiency = total_move.div(total_path_60d.replace(0, np.nan))
    trend_quality = (0.50 * robust_z_by_day(cum_tsi.abs()) +
                     0.30 * robust_z_by_day(rho.clip(lower=0)) +
                     0.20 * robust_z_by_day(trend_efficiency))
    liquidity = np.log1p(
        median_value_60d.where(median_value_60d > 0,
                               median_volume_60d).fillna(0.0))

    score = (0.22 * robust_z_by_day(avg_amp_pct) +
             0.20 * robust_z_by_day(atr_pct) +
             0.16 * robust_z_by_day(realized_vol_ann_pct) +
             0.08 * robust_z_by_day(p80_amp_pct) +
             0.14 * robust_z_by_day(opportunity_after_cost_pct) +
             0.15 * trend_quality + 0.05 * robust_z_by_day(liquidity) -
             0.20 * jump_share_60d.clip(0, 1))
    score = score.where(roundtrip_cost_60d.notna())
    score_rank = score.rank(axis=1, ascending=False, method="first")
    if min_active_days is None:
        min_active_days = max(1, int(window_days * 0.6))
    valid = ((rows_60d >= min_rows) & (active_days_60d >= min_active_days)
             & roundtrip_cost_60d.notna())

    if high_vol_top_pct is not None:
        daily_count = score.notna().sum(axis=1).clip(lower=1)
        daily_top_n = np.ceil(daily_count * high_vol_top_pct).clip(lower=1)
        is_high_vol = score_rank.le(daily_top_n, axis=0) & valid
    else:
        is_high_vol = score_rank.le(high_vol_top_n) & valid

    if use_abs_threshold:
        is_high_vol = (is_high_vol & (avg_amp_pct >= min_amp_pct) &
                       (atr_pct >= min_atr_pct) &
                       (amp_to_cost >= min_amp_to_cost) & (rho >= min_rho) &
                       (cum_tsi.abs() >= min_abs_cum_tsi))
    out = pd.concat(
        {
            "score": score,
            "rank": score_rank,
            "is_high_vol": is_high_vol,
            "rows": rows_60d,
            "active_days": active_days_60d,
            "first_close": daily_first,
            "last_close": daily_last,
            "avg_amp_pct": avg_amp_pct,
            "p80_amp_pct": p80_amp_pct,
            "atr_pct": atr_pct,
            "realized_vol_ann_pct": realized_vol_ann_pct,
            "tsi": tsi,
            "cum_tsi": cum_tsi,
            "rho": rho,
            "trend_quality": trend_quality,
            "roundtrip_cost_bp": roundtrip_cost_bp,
            "amp_to_cost": amp_to_cost,
            "opportunity_after_cost_pct": opportunity_after_cost_pct,
            "trend_efficiency": trend_efficiency,
            "jump_share": jump_share_60d,
            "max_drawdown": rolling_max_drawdown,
            "median_volume": median_volume_60d,
            "median_value": median_value_60d,
            "median_openint": median_openint_60d
        },
        axis=1)
    return out
