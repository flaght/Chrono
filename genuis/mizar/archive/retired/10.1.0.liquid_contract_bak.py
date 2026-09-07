import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Iterable, List
from dotenv import load_dotenv

load_dotenv()
from kdutils.data import *
from ultron.tradingday import *
from ultron.sentry.api import *

codes1 = [
    'RB', 'J', 'ZN', 'AU', 'AG', 'SC', 'RM', 'RU', 'FG', 'SA', 'I', 'JM', 'HC',
    'SF', 'SM', 'TA', 'MA', 'EG', 'L', 'PP', 'V', 'FU', 'BU', 'A', 'Y', 'M',
    'OI', 'P', 'C', 'CS', 'SR', 'CF', 'JD', 'NI', 'AL', 'CU', 'PB'
]

OUTPUT_COLUMNS = [
    "code",
    "score",
    "is_high_vol",
    "rows",
    "active_days",
    "start",
    "end",
    "avg_amp_pct",
    "p80_amp_pct",
    "atr_pct",
    "realized_vol_ann_pct",
    "roundtrip_cost_bp",
    "amp_to_cost",
    "opportunity_after_cost_pct",
    "trend_efficiency",
    "jump_share",
    "median_volume",
    "median_value",
    "median_openint",
]


def diagnose_missing_by_trade_day(df1: pd.DataFrame,
                                  fields=None,
                                  top_n=50) -> pd.DataFrame:
    """
    诊断宽表 df1 在哪些 trade_date/code/field 上存在空值。

    df1:
        market_data.set_index(['trade_time', 'code']).unstack() 的结果

    返回字段:
        trade_date, code, field, missing, total, missing_pct
    """
    if fields is None:
        fields = [
            'open', 'high', 'low', 'close', 'volume', 'value', 'openint',
            'tradeCommiNum'
        ]

    trade_day = _trade_day_index(df1['close'].index)
    rows = []
    for field in fields:
        if field not in df1.columns.get_level_values(0):
            continue
        frame = df1[field]
        missing = frame.isna().groupby(trade_day).sum()
        total = frame.groupby(trade_day).size()
        total = pd.DataFrame(np.repeat(total.values[:, None],
                                       len(frame.columns),
                                       axis=1),
                             index=total.index,
                             columns=frame.columns)
        hit = missing[missing > 0].stack()
        if hit.empty:
            continue
        part = hit.rename('missing').reset_index()
        part = part.rename(columns={'level_1': 'code'})
        part['field'] = field
        part['total'] = [
            total.loc[d, c] for d, c in zip(part['trade_date'], part['code'])
        ]
        part['missing_pct'] = part['missing'] / part['total']
        rows.append(part)

    if not rows:
        return pd.DataFrame(columns=[
            'trade_date', 'code', 'field', 'missing', 'total', 'missing_pct'
        ])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(['missing_pct', 'missing'], ascending=[False, False])
    return out.head(top_n)


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


def _interval_trading_date(begin_date: datetime.date, end_date: datetime):
    dates = makeSchedule(begin_date,
                         end_date,
                         '1b',
                         calendar='china.sse',
                         dateRule=BizDayConventions.Following,
                         dateGenerationRule=DateGeneration.Backward)
    return dates


def _previous_trading_date(trading_date: datetime.date) -> datetime.date:
    previous_day = advanceDateByCalendar("china.sse", trading_date, "-1b")
    if isinstance(previous_day, datetime.datetime):
        return previous_day.date()
    if isinstance(previous_day, pd.Timestamp):
        return previous_day.date()
    return previous_day


def _robust_z_by_day(frame: pd.DataFrame) -> pd.DataFrame:
    """对每个交易日横截面做 robust z。"""
    return frame.apply(_robust_z, axis=1)


def _trade_day_index(
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
            d: _previous_trading_date(d)
            for d in trade_day[mask].unique()
        }
        trade_day.loc[mask] = trade_day.loc[mask].map(prev_map)

    return pd.Index(pd.to_datetime(trade_day), name="trade_date")


def start2(slippage_bp=0.001,
           window_days=60,
           min_periods=30,
           min_rows=60,
           min_amp_pct=0.1,
           min_atr_pct=0.1,
           min_amp_to_cost=1.0,
           high_vol_top_n=30,
           high_vol_top_pct=None,
           adjusted_method='pcr'):
    end_time = datetime.datetime(2026, 6, 5)
    begin_time = advanceDateByCalendar('china.sse', end_time, '-90b')
    start_time = advanceDateByCalendar('china.sse', begin_time, '-1b')
    pdb.set_trace()
    basic_infos = fetch_basic2(
        begin_date=advanceDateByCalendar('china.sse', begin_time, '-60b'),
        end_date=advanceDateByCalendar('china.sse', end_time, '90b'),
        codes=codes1,  #[0:5],
        columns=[
            'contractObject', 'code', 'exchangeCD', 'contMultNum',
            'lastTradeDate', 'tradeCommiNum'
        ])
    basic_infos['tradeCommiNum'] = basic_infos['tradeCommiNum'] / 100

    codes = basic_infos['code'].unique().tolist()

    # market_data = fetch_main_market(begin_date=start_time,
    #                                 end_date=end_time,
    #                                 codes=codes,
    #                                 method=adjusted_method,
    #                                 keep_symbol=True)

    # end_time = datetime.datetime(2026, 6, 5)
    # start_time = datetime.datetime(2026, 6, 1)
    market_data = fetch_local_market1(base_path=os.environ['BAR_FUT_DIRS'],
                                      begin_date=start_time,
                                      end_date=end_time,
                                      codes=codes,
                                      method=adjusted_method,
                                      keep_symbol=True)
    pdb.set_trace()
    market_data = market_data[(market_data['trade_time'] > begin_time)
                              & (market_data['trade_time'] <= end_time)]

    market_data = market_data.sort_values(by=['trade_time', 'code']).merge(
        basic_infos, on=['code', 'symbol'], how='left')
    dates = _interval_trading_date(begin_date=market_data['trade_time'].min(),
                                   end_date=market_data['trade_time'].max())
    market_data = market_data.drop_duplicates(subset=['trade_time', 'code'])
    market_data = market_data[pd.to_datetime(
        market_data['trade_time']).dt.normalize().isin(dates)]
    df1 = market_data.set_index(['trade_time', 'code']).unstack()

    prev_close = df1['close'].shift(1)
    ratio = (df1["close"] / prev_close).where(prev_close > 0)
    amp = (df1["high"] - df1["low"]) / df1["close"]

    tr = pd.concat(
        [
            df1["high"] - df1["low"],
            (df1["high"] - prev_close).abs(),
            (df1["low"] - prev_close).abs(),
        ],
        axis=0,
        keys=["hl", "hc", "lc"],
    ).groupby(level=1).max()

    atr_part = tr / df1['close']
    log_ret = np.log(ratio.where(ratio > 0))
    abs_log_ret = log_ret.abs()

    default_cost = basic_infos.groupby(
        'code')['tradeCommiNum'].median() + slippage_bp
    roundtrip_cost = df1['tradeCommiNum'].add(slippage_bp)
    roundtrip_cost = roundtrip_cost.fillna(default_cost)

    opportunity = (amp - roundtrip_cost).clip(lower=0.0)
    jump_flag = (atr_part > amp * 2.5).astype(float)
    pdb.set_trace()

    trade_day = _trade_day_index(df1['close'].index)
    daily_rows = df1['close'].notna().astype("int64").groupby(trade_day).sum()
    rows_60d = daily_rows.rolling(window_days, min_periods=min_periods).sum()

    daily_active = df1['close'].notna().groupby(trade_day).any().astype(
        "int64")
    active_days_60d = daily_active.rolling(window_days,
                                           min_periods=min_periods).sum()

    daily_first = df1['close'].groupby(trade_day).first().bfill(
        limit=window_days - 1).shift(window_days - 1)
    daily_last = df1['close'].groupby(trade_day).last().ffill(
        limit=window_days - 1)

    rolling_peak = daily_last.rolling(window_days,
                                      min_periods=min_periods).max()
    daily_dd = daily_last.div(rolling_peak) - 1.0
    max_drawdown = daily_dd.rolling(window_days, min_periods=min_periods).min()

    daily_amp_sum = amp.groupby(trade_day).sum()
    daily_amp_p80 = amp.groupby(trade_day).quantile(0.80)
    daily_atr_sum = atr_part.groupby(trade_day).sum()
    daily_ret_sum = log_ret.groupby(trade_day).sum()
    daily_ret_sumsq = log_ret.pow(2).groupby(trade_day).sum()
    daily_ret_count = log_ret.notna().astype("int64").groupby(trade_day).sum()
    daily_path_sum = abs_log_ret.groupby(trade_day).sum()
    daily_opp_sum = opportunity.groupby(trade_day).sum()
    daily_jump_sum = jump_flag.groupby(trade_day).sum()

    avg_amp_60d = daily_amp_sum.rolling(
        window_days, min_periods=min_periods).sum() / rows_60d
    p80_amp_60d = daily_amp_p80.rolling(window_days,
                                        min_periods=min_periods).mean()
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

    median_volume_60d = df1['volume'].groupby(trade_day).median().rolling(
        window_days, min_periods=min_periods).median()
    median_value_60d = df1['value'].groupby(trade_day).median().rolling(
        window_days, min_periods=min_periods).median()
    median_openint_60d = df1['openint'].groupby(trade_day).median().rolling(
        window_days, min_periods=min_periods).median()
    roundtrip_cost_60d = roundtrip_cost.groupby(trade_day).median().rolling(
        window_days, min_periods=min_periods).median()

    roundtrip_cost_daily = roundtrip_cost.groupby(trade_day).median()
    roundtrip_cost_60d = roundtrip_cost_daily.ffill().fillna(default_cost)

    periods_per_year = 252.0 * rows_60d.div(
        active_days_60d.clip(lower=1)).clip(lower=1.0)
    total_move = np.log(
        daily_last.div(daily_first).where(daily_first > 0)).abs()

    avg_amp_pct = avg_amp_60d * 100.0
    p80_amp_pct = p80_amp_60d * 100.0
    atr_pct = atr_60d * 100.0
    realized_vol_ann_pct = ret_std_60d.fillna(0.0) * np.sqrt(
        periods_per_year) * 100.0
    roundtrip_cost_bp = roundtrip_cost_60d * 10000.0
    amp_to_cost = avg_amp_60d.div(roundtrip_cost_60d.replace(0, np.nan))
    opportunity_after_cost_pct = opportunity_60d * 100.0
    trend_efficiency = total_move.div(total_path_60d.replace(0, np.nan))
    liquidity = np.log1p(
        median_value_60d.where(median_value_60d > 0,
                               median_volume_60d).fillna(0.0))

    score = (0.30 * _robust_z_by_day(avg_amp_pct) +
             0.25 * _robust_z_by_day(atr_pct) +
             0.20 * _robust_z_by_day(realized_vol_ann_pct) +
             0.10 * _robust_z_by_day(p80_amp_pct) +
             0.10 * _robust_z_by_day(opportunity_after_cost_pct) +
             0.05 * _robust_z_by_day(liquidity) -
             0.20 * jump_share_60d.clip(0, 1))
    score_rank = score.rank(axis=1, ascending=False, method="first")
    valid = rows_60d >= min_rows

    if high_vol_top_pct is not None:
        daily_count = score.notna().sum(axis=1).clip(lower=1)
        daily_top_n = np.ceil(daily_count * high_vol_top_pct).clip(lower=1)
        is_high_vol = score_rank.le(daily_top_n, axis=0) & valid
    else:
        is_high_vol = score_rank.le(high_vol_top_n) & valid

    is_high_vol = (is_high_vol & (avg_amp_pct >= min_amp_pct) &
                   (atr_pct >= min_atr_pct) & (amp_to_cost >= min_amp_to_cost))
    out = pd.concat(
        {
            "score": score,  # 综合评分
            "rank": score_rank,
            "is_high_vol": is_high_vol,
            "rows": rows_60d,
            "active_days":
            active_days_60d,  # 过去 window_days 内有效分钟 K 数量。防止数据太少导致指标不可靠。
            "first_close": daily_first,  #窗口内第一个有效收盘价。用于计算窗口净位移。
            "last_close": daily_last,  # 窗口内最后一个有效收盘价。和 first_close 一起计算趋势效率。
            "avg_amp_pct": avg_amp_pct,  # 平均分钟振幅 衡量日内每根 K 线平均能给多少价格空间。
            "p80_amp_pct":
            p80_amp_pct,  # 分钟振幅的 80 分位。比均值更关注“较活跃时”的波动水平，避免被大量平淡分钟压低。
            "atr_pct":
            atr_pct,  # TR 会考虑上一分钟 close 到当前 high/low 的跳动，比单纯 high-low 更完整。
            "realized_vol_ann_pct":
            realized_vol_ann_pct,  # 基于分钟收益率标准差年化后的实现波动率。衡量价格变化强度和频率。
            "roundtrip_cost_bp": roundtrip_cost_bp,  # 往返交易成本，单位 bp。包含开平仓手续费和滑点
            "amp_to_cost": amp_to_cost,  # 平均振幅 / 往返成本。越高越好，表示波动空间能覆盖成本的倍数。
            "opportunity_after_cost_pct":
            opportunity_after_cost_pct,  # 扣除成本后的平均可用空间 用于判断“看起来有波动”是否真的能交易。
            "trend_efficiency": trend_efficiency,  # 越高说明窗口内更偏单边趋势；越低说明来回震荡多。
            "jump_share":
            jump_share_60d,  # 疑似跳跃波动占比 过高说明波动可能来自跳空/断点，不一定适合日内连续交易。
            "max_drawdown": max_drawdown,  # 窗口内最大回撤。用于识别品种在窗口内的下行风险或单边回撤压力。
            "median_volume": median_volume_60d,  # 窗口内成交量中位数。衡量交易活跃度。
            "median_value": median_value_60d,  # 窗口内成交额中位数。比成交量更适合跨品种比较。
            "median_openint": median_openint_60d  # 窗口内持仓量中位数。衡量合约承载资金和主力稳定性。
        },
        axis=1)
    pdb.set_trace()
    print('-->')


def start1(slippage_bp=0.001, min_rows=60, adjusted_method='pcr'):
    pdb.set_trace()
    end_time = datetime.datetime(2026, 6, 5)
    begin_time = advanceDateByCalendar('china.sse', end_time, '-60b')

    basic_infos = fetch_basic2(
        begin_date=advanceDateByCalendar('china.sse', begin_time, '-60b'),
        end_date=advanceDateByCalendar('china.sse', end_time, '120b'),
        codes=codes1[0:5],
        columns=[
            'contractObject', 'code', 'exchangeCD', 'contMultNum',
            'lastTradeDate', 'tradeCommiNum'
        ])
    basic_infos['tradeCommiNum'] = basic_infos['tradeCommiNum'] / 100

    codes = basic_infos['code'].unique().tolist()

    # market_data = fetch_main_market(begin_date=begin_time,
    #                                 end_date=end_time,
    #                                 codes=codes,
    #                                 method=adjusted_method,
    #                                 keep_symbol=True)

    market_data = fetch_local_market1(base_path=os.environ['BAR_FUT_DIRS'],
                                      begin_date=begin_time,
                                      end_date=end_time,
                                      codes=codes,
                                      method=adjusted_method,
                                      keep_symbol=True)

    market_data = market_data.sort_values(by=['trade_time', 'code']).merge(
        basic_infos, on=['code', 'symbol'], how='left')
    pdb.set_trace()
    df1 = market_data.set_index(['trade_time', 'code']).unstack()

    pdb.set_trace()
    prev_close = df1['close'].shift(1)
    ratio = (df1["close"] / prev_close).where(prev_close > 0)
    amp = (df1["high"] - df1["low"]) / df1["close"]

    tr = pd.concat(
        [
            df1["high"] - df1["low"],
            (df1["high"] - prev_close).abs(),
            (df1["low"] - prev_close).abs(),
        ],
        axis=0,
        keys=["hl", "hc", "lc"],
    ).groupby(level=1).max()

    atr_part = tr / df1['close']
    log_ret = np.log(ratio.where(ratio > 0))
    abs_log_ret = log_ret.abs()

    pdb.set_trace()
    roundtrip_cost = df1['tradeCommiNum'] + slippage_bp
    opportunity = (amp - roundtrip_cost).clip(lower=0.0)
    jump_flag = (atr_part > amp * 2.5).astype(float)

    rows = df1['close'].count()
    active_days = df1['close'].notna().groupby(
        df1['close'].index.normalize()).any().sum()
    first_close = df1['close'].bfill().iloc[0]
    last_close = df1['close'].ffill().iloc[-1]
    equity_curve = df1['close'] / first_close
    drawdown = equity_curve / equity_curve.cummax() - 1.0

    out = pd.DataFrame({
        "rows":
        rows,
        "active_days":
        active_days,
        "start":
        df1['close'].apply(lambda s: s.first_valid_index()),
        "end":
        df1['close'].apply(lambda s: s.last_valid_index()),
        "first_close":
        first_close,
        "last_close":
        last_close,
        "avg_amp":
        amp.mean(),
        "p80_amp":
        amp.quantile(0.80),
        "atr":
        atr_part.mean(),
        "ret_std":
        log_ret.std(),
        "total_path":
        abs_log_ret.sum(),
        "opportunity":
        opportunity.mean(),
        "jump_share":
        jump_flag.mean(),
        "max_drawdown":
        drawdown.min(),
        "median_volume":
        df1['volume'].median(),
        "median_value":
        df1['value'].median(),
        "median_openint":
        df1['openint'].median(),
        "roundtrip_cost":
        roundtrip_cost.median()
    })
    pdb.set_trace()

    out = out[out["rows"] >= min_rows].copy()
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    periods_per_year = 252.0 * (
        out["rows"] / out["active_days"].clip(lower=1)).clip(lower=1.0)
    total_move = np.log(
        (out["last_close"] /
         out["first_close"]).where(out["first_close"] > 0)).abs()

    out["avg_amp_pct"] = out["avg_amp"] * 100.0
    out["p80_amp_pct"] = out["p80_amp"] * 100.0
    out["atr_pct"] = out["atr"] * 100.0
    out["realized_vol_ann_pct"] = out["ret_std"].fillna(0.0) * np.sqrt(
        periods_per_year) * 100.0
    out["roundtrip_cost_bp"] = out["roundtrip_cost"] * 10000.0
    out["amp_to_cost"] = out["avg_amp"] / out["roundtrip_cost"].replace(
        0, np.nan)
    out["amp_to_cost"] = out["amp_to_cost"].replace([np.inf, -np.inf],
                                                    np.nan).fillna(np.inf)
    out["opportunity_after_cost_pct"] = out["opportunity"] * 100.0
    out["trend_efficiency"] = (
        total_move / out["total_path"].replace(0, np.nan)).fillna(0.0)

    out['liquidity'] = np.log1p(out["median_value"].where(
        out["median_value"] > 0, out["median_volume"]).fillna(0.0))
    pdb.set_trace()

    out["score"] = (0.30 * _robust_z(out["avg_amp_pct"]) +
                    0.25 * _robust_z(out["atr_pct"]) +
                    0.20 * _robust_z(out["realized_vol_ann_pct"]) +
                    0.10 * _robust_z(out["p80_amp_pct"]) +
                    0.10 * _robust_z(out["opportunity_after_cost_pct"]) +
                    0.05 * _robust_z(out["liquidity"]) -
                    0.20 * out["jump_share"].clip(0, 1))
    out["is_high_vol"] = ((out["avg_amp_pct"] >= min_amp_pct)
                          & (out["atr_pct"] >= min_atr_pct)
                          & (out["amp_to_cost"] >= min_amp_to_cost))

    out = out.reset_index(names="code")
    out["start"] = out["start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["end"] = out["end"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out.sort_values(["is_high_vol", "score"], ascending=[False, False])


if __name__ == '__main__':
    start2()
