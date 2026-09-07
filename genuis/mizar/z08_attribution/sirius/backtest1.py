import datetime, itertools
from collections import namedtuple
from dotenv import load_dotenv

load_dotenv()

from lib.rl012.sandbox import PositionBacktester
from ultron.tradingday import *
from chaosmind.timing.sirius0002.workflow import WorkFlow
from lib.uvx import load_sirius_params
from lib.attr001.ftd001 import *

import pandas as pd


def build_position_signals(model_output: pd.DataFrame,
                           hold_bars: int = 15,
                           max_position: int = 10,
                           lot_per_signal: int = 1,
                           cooldown_bars: int = 0,
                           prevent_same_direction_reentry: bool = True):
    """
    根据离散信号列 `signal` 构造开平仓指令。

    输入要求:
    - model_output 必须包含:
      - code
      - trade_time
      - value   : 连续值，仅用于记录 source_value
      - signal  : 离散信号，取值为 -1 / 0 / 1

    逻辑:
    - signal = 1  -> 开多
    - signal = -1 -> 开空
    - signal = 0  -> 不开仓
    - 开仓后固定持有 hold_bars 根 bar
    - 到期自动反向生成平仓信号
    - 同时持仓总手数不超过 max_position
    - 若 prevent_same_direction_reentry=True，则已有同方向持仓时不重复开仓
    - 若 cooldown_bars > 0，则同方向开仓后需要等待 cooldown_bars 根 bar 才允许再次开仓
    """
    required_cols = {"code", "trade_time", "value", "signal"}
    missing = required_cols - set(model_output.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if hold_bars <= 0:
        raise ValueError("hold_bars must be > 0")
    if max_position <= 0:
        raise ValueError("max_position must be > 0")
    if lot_per_signal <= 0:
        raise ValueError("lot_per_signal must be > 0")
    if cooldown_bars < 0:
        raise ValueError("cooldown_bars must be >= 0")

    df = model_output.copy()
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df = df.sort_values(["code", "trade_time"]).reset_index(drop=True)

    records = []
    pair_id = 0

    for code, group in df.groupby("code", sort=True):
        group = group.sort_values("trade_time").reset_index(drop=True)

        active_positions = []
        active_lots = 0
        last_open_idx_by_direction = {1: None, -1: None}

        for i, row in group.iterrows():
            current_time = row["trade_time"]

            # 1. 先处理当前 bar 到期的平仓，这样同一根 bar 上的后续开仓判断
            #    是基于“已经释放掉到期仓位”的真实状态。
            remaining = []
            for pos in active_positions:
                if pos["close_idx"] == i:
                    records.append({
                        "date": current_time.normalize(),
                        "min_time": current_time.strftime("%H%M"),
                        "code": code,
                        "direction": -pos["direction"],
                        "numbers": pos["numbers"],
                        "signal_type": "close",
                        "pair_id": pos["pair_id"],
                        "source_value": pos["source_value"],
                        "open_trade_time": pos["open_trade_time"],
                        "close_trade_time": pos["close_trade_time"],
                    })
                    active_lots -= pos["numbers"]
                else:
                    remaining.append(pos)
            active_positions = remaining

            # 2. 只使用离散 signal，不再基于 value 再做一次方向判断。
            signal = int(row["signal"])
            if signal == 0:
                continue
            if signal not in (-1, 1):
                raise ValueError(f"signal must be in {{-1,0,1}}, got {signal}")

            # 3. 若已有同方向持仓，则跳过，避免同方向信号连续出现时不断滚动加仓。
            if prevent_same_direction_reentry:
                has_same_direction = any(pos["direction"] == signal
                                         for pos in active_positions)
                if has_same_direction:
                    continue

            # 4. 同方向 cooldown。即使旧仓已经平掉，也要求等待若干 bar 后才能再次开同方向。
            if cooldown_bars > 0:
                last_open_idx = last_open_idx_by_direction[signal]
                if last_open_idx is not None and (
                        i - last_open_idx) < cooldown_bars:
                    continue

            # 5. 仓位上限控制。限制的是同时持有的总手数，而不是一天交易次数。
            if active_lots + lot_per_signal > max_position:
                continue

            # 6. 若剩余 bar 不足持有期，则不再开仓，避免生成无法按规则平掉的尾部仓位。
            close_idx = i + hold_bars
            if close_idx >= len(group):
                continue

            open_direction = signal
            close_time = group.iloc[close_idx]["trade_time"]
            source_value = float(row["value"])

            pair_id += 1
            records.append({
                "date": current_time.normalize(),
                "min_time": current_time.strftime("%H%M"),
                "code": code,
                "direction": open_direction,
                "numbers": lot_per_signal,
                "signal_type": "open",
                "pair_id": pair_id,
                "source_value": source_value,
                "open_trade_time": current_time,
                "close_trade_time": close_time,
            })

            active_positions.append({
                "direction": open_direction,
                "numbers": lot_per_signal,
                "close_idx": close_idx,
                "pair_id": pair_id,
                "source_value": source_value,
                "open_trade_time": current_time,
                "close_trade_time": close_time,
            })
            active_lots += lot_per_signal
            last_open_idx_by_direction[signal] = i

    position_df = pd.DataFrame(records)
    if position_df.empty:
        return pd.DataFrame(columns=[
            "date", "min_time", "code", "direction", "numbers", "pair_id",
            "source_value", "open_trade_time", "close_trade_time"
        ])

    position_df["signal_trade_time"] = pd.to_datetime(
        position_df["date"].astype(str) + " " +
        position_df["min_time"].str.slice(0, 2) + ":" +
        position_df["min_time"].str.slice(2, 4) + ":00")

    # 同一时间戳下，先执行 close 再执行 open，和上面的生成逻辑保持一致。
    position_df["signal_type_order"] = position_df["signal_type"].map({
        "close":
        0,
        "open":
        1,
    }).fillna(9)

    position_df = position_df.sort_values(
        ["code", "signal_trade_time", "signal_type_order",
         "pair_id"]).reset_index(drop=True)

    position_df = position_df.drop(
        columns=["signal_trade_time", "signal_type_order", "signal_type"])
    return position_df


def load_factors_data(factors_infos, category, instruments, start_time,
                      end_time1):
    names = [f['formula'] for f in factors_infos]
    factors_data = fetch_metrics(
        category=category,
        code=INSTRUMENTS_CODES[instruments],
        begin_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end_time1.strftime("%Y-%m-%d %H:%M:%S"),
        names=names,
        table_name='raw_factors')
    return factors_data


def load_market_data(instruments, begin_time, end_time, trading_sessions):
    market_data = fetch_bench_data(instruments=instruments,
                                   begin_time=begin_time,
                                   end_time=end_time,
                                   adjusted_method=None)

    market_data = filter_trading_time(data=market_data,
                                      trading_sessions=trading_sessions)
    return market_data


def start2(instruments, task_id):
    begin_time = datetime.datetime(2026, 4, 1)
    end_time = datetime.datetime(2026, 5, 25)

    start_time = advanceDateByCalendar('china.sse', begin_time, '-1b')
    end_time1 = advanceDateByCalendar('china.sse', end_time, '1b')

    trading_sessions = (("21:00", "23:00"), ("09:00", "10:15"),
                        ("10:30", "11:30"), ("13:30", "15:00"))

    filename = "{0}_{1}_beacktest_values.feather".format(instruments, task_id)
    actions = pd.read_feather(filename)

    market_data = load_market_data(instruments=instruments,
                                   begin_time=begin_time,
                                   end_time=end_time1,
                                   trading_sessions=trading_sessions)

    factors_infos, params = load_sirius_params(
        code=INSTRUMENTS_CODES[instruments], task_id=task_id)

    workflow = WorkFlow(directory=params['model_path'],
                        code=INSTRUMENTS_CODES[instruments],
                        symbol='rb9999',
                        task_id=task_id,
                        factors_infos=factors_infos,
                        softmax_temperature=params['softmax_temperature'],
                        min_open_signal_abs=params['min_open_signal_abs'],
                        logit_clip=params['logit_clip'],
                        min_trade_advantage=params['min_trade_advantage'],
                        min_margin=params['min_margin'],
                        method=params['method'],
                        win=params['win'])

    res = []
    for row in actions.itertuples():
        print(row.trade_time)
        rt1 = workflow.conversion_signals(trade_time=row.trade_time,
                                          raw_action=row.value)
        res.append(rt1)

    signals_df = pd.DataFrame(res)
    signals_df['trade_date'] = pd.to_datetime(
        signals_df['trade_time']).dt.strftime("%Y-%m-%d")
    position_data = build_position_signals(model_output=signals_df,
                                           hold_bars=5,
                                           max_position=1,
                                           lot_per_signal=1)
    pb = PositionBacktester(market_data=market_data,
                            contract_multiplier=10,
                            slippage=0.1)
    trade_records, daily_stats = pb.run(position_df=position_data, code='RB')


def start1(instruments, task_id):
    category = 'bench'

    begin_time = datetime.datetime(2026, 4, 1)
    end_time = datetime.datetime(2026, 5, 26)

    start_time = advanceDateByCalendar('china.sse', begin_time, '-1b')
    end_time1 = advanceDateByCalendar('china.sse', end_time, '1b')

    factors_infos, params = load_sirius_params(
        code=INSTRUMENTS_CODES[instruments], task_id=task_id)
    workflow = WorkFlow(directory=params['model_path'],
                        code=INSTRUMENTS_CODES[instruments],
                        symbol='rb9999',
                        task_id=task_id,
                        factors_infos=factors_infos,
                        softmax_temperature=params['softmax_temperature'],
                        min_open_signal_abs=params['min_open_signal_abs'],
                        logit_clip=params['logit_clip'],
                        min_trade_advantage=params['min_trade_advantage'],
                        min_margin=params['min_margin'],
                        method=params['method'],
                        win=params['win'])
    pdb.set_trace()
    factors_data = load_factors_data(factors_infos=factors_infos,
                                     category=category,
                                     instruments=instruments,
                                     start_time=start_time,
                                     end_time1=end_time1)
    factors_data = factors_data.pivot_table(index=['trade_time', 'code'],
                                            columns='name',
                                            values='value',
                                            aggfunc='last')

    total_data1 = factors_data.dropna()
    all_trade_times = total_data1.index.get_level_values(
        'trade_time').unique().sort_values()
    res = []
    for time in all_trade_times:
        action = workflow.create_values(trade_time=time,
                                        data=total_data1,
                                        deterministic=True)
        res.append(action)
    pd.DataFrame(res).to_feather("{0}_{1}_beacktest_values.feather".format(
        instruments, task_id))


if __name__ == '__main__':
    start1(instruments='rbb', task_id='1029921127239410')
    start2(instruments='rbb', task_id='1029921127239410')
