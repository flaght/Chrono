import pandas as pd
from pathlib import Path
from lumina.genetic.signal.method import *
from lib.attr001.ftd001 import *
from kdutils.macro2 import *


### 转信号
def create_signal(data, signal_method, signal_params):
    pos_data = eval(signal_method)(factor_data=data.set_index(
        ['trade_time', 'code'])[['transformed']],
                                   **signal_params)
    pos_data = pos_data.stack()
    pos_data.name = 'signal'
    return pos_data.reset_index()


def build_locked_signals(model_output: pd.DataFrame,
                         base_position: int = 10,
                         lot_per_signal: int = 1,
                         cooldown_bars: int = 0,
                         hold_bars: int = None,
                         entry_resampling_win: int = None,
                         date_col: str = None) -> pd.DataFrame:
    """
    将离散 signal 转成对锁底仓回测使用的“平底仓事件”。

    这个函数用于 sandbox.PositionBacktester，语义和原始
    backtest/trade_backtest.ipynb 保持一致:
    - signal = 1  -> direction = 1，买入平空，消耗 1 手空头底仓
    - signal = -1 -> direction = -1，卖出平多，消耗 1 手多头底仓
    - signal = 0  -> 不生成交易事件

    注意:
    - hold_bars=None 时，不生成恢复对锁信号，持有到 sandbox 的当日收盘结算。
    - hold_bars>0 时，信号触发后持有 hold_bars 根 bar，再生成反向事件恢复对锁。
    - entry_resampling_win>1 时，只在 minute % entry_resampling_win == 0 的
      时间点接受开暴露信号，用于对齐 FactorEvaluate1 的 resampling_win。
    - 每个交易日每个方向最多消耗 base_position 手底仓。
    - sandbox 会在每日开盘自动恢复对锁底仓，并按当日收盘价结算。
    """
    required_cols = {"code", "trade_time", "signal"}
    missing = required_cols - set(model_output.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if base_position <= 0:
        raise ValueError("base_position must be > 0")
    if lot_per_signal <= 0:
        raise ValueError("lot_per_signal must be > 0")
    if cooldown_bars < 0:
        raise ValueError("cooldown_bars must be >= 0")
    if hold_bars is not None and hold_bars <= 0:
        raise ValueError("hold_bars must be > 0 when provided")
    if entry_resampling_win is not None and entry_resampling_win <= 0:
        raise ValueError("entry_resampling_win must be > 0 when provided")

    df = model_output.copy()
    df["trade_time"] = pd.to_datetime(df["trade_time"])

    if date_col is not None:
        if date_col not in df.columns:
            raise ValueError(
                f"date_col={date_col!r} not found in model_output")
        df["_signal_date"] = pd.to_datetime(df[date_col]).dt.normalize()
    else:
        df["_signal_date"] = df["trade_time"].dt.normalize()

    df["_min_time"] = df["trade_time"].dt.strftime("%H%M")
    df = df.sort_values(["code", "_signal_date",
                         "trade_time"]).reset_index(drop=True)

    records = []
    value_col = "value" if "value" in df.columns else None

    for (code, signal_date), group in df.groupby(["code", "_signal_date"],
                                                 sort=True):
        remaining_by_direction = {1: base_position, -1: base_position}
        last_accept_idx_by_direction = {1: None, -1: None}
        restore_events_by_idx = {}

        signals = group["signal"].to_numpy()
        trade_times = group["trade_time"].to_numpy()
        min_times = group["_min_time"].to_numpy()
        minutes = group["trade_time"].dt.minute.to_numpy()
        values = (group[value_col].to_numpy(dtype=float, copy=False)
                  if value_col else None)

        for i, raw_signal in enumerate(signals):
            restore_events = restore_events_by_idx.pop(i, None)
            if restore_events:
                for event in restore_events:
                    records.append(event["record"])
                    remaining_by_direction[
                        event["opened_direction"]] += event["numbers"]

            signal = int(raw_signal)
            if signal == 0:
                continue
            if signal not in (-1, 1):
                raise ValueError(f"signal must be in {{-1,0,1}}, got {signal}")
            if entry_resampling_win and entry_resampling_win > 1:
                if int(minutes[i]) % int(entry_resampling_win) != 0:
                    continue

            close_idx = None
            if hold_bars is not None:
                close_idx = i + hold_bars
                if close_idx >= len(group):
                    continue

            if cooldown_bars > 0:
                last_idx = last_accept_idx_by_direction[signal]
                if last_idx is not None and (i - last_idx) < cooldown_bars:
                    continue

            remaining = remaining_by_direction[signal]
            if remaining <= 0:
                continue

            numbers = min(lot_per_signal, remaining)
            remaining_by_direction[signal] -= numbers
            last_accept_idx_by_direction[signal] = i

            record = {
                "date":
                pd.Timestamp(signal_date),
                "min_time":
                str(min_times[i]),
                "code":
                code,
                "direction":
                signal,
                "numbers":
                int(numbers),
                "trade_time":
                pd.Timestamp(trade_times[i]),
                "remaining_same_side":
                int(remaining_by_direction[signal]),
                "signal_type":
                "open_exposure" if hold_bars is not None else "consume_lock",
            }
            if values is not None:
                record["source_value"] = float(values[i])
            records.append(record)

            if hold_bars is not None:
                restore_record = {
                    "date": pd.Timestamp(signal_date),
                    "min_time": str(min_times[close_idx]),
                    "code": code,
                    "direction": -signal,
                    "numbers": int(numbers),
                    "trade_time": pd.Timestamp(trade_times[close_idx]),
                    "remaining_same_side": int(remaining_by_direction[signal]),
                    "signal_type": "restore_lock",
                }
                if values is not None:
                    restore_record["source_value"] = float(values[i])
                restore_events_by_idx.setdefault(close_idx, []).append({
                    "opened_direction":
                    signal,
                    "numbers":
                    int(numbers),
                    "record":
                    restore_record,
                })

    if not records:
        columns = [
            "date", "min_time", "code", "direction", "numbers", "trade_time",
            "remaining_same_side", "signal_type"
        ]
        if value_col:
            columns.append("source_value")
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records).sort_values(["code", "date", "min_time"
                                              ]).reset_index(drop=True)


def build_capped_locked_signals(
        model_output: pd.DataFrame,
        base_position: int = 10,
        lot_per_signal: int = 1,
        cooldown_bars: int = 0,
        hold_bars: int = None,
        entry_resampling_win: int = None,
        date_col: str = None,
        max_daily_open_lots: int = None,
        max_daily_open_lots_per_direction: int = None,
        max_active_open_lots: int = None,
        max_active_open_lots_per_direction: int = None,
        min_abs_value: float = None,
        block_same_direction_reentry: bool = False,
        block_opposite_direction_reentry: bool = False) -> pd.DataFrame:
    """
    将离散 signal 转成对锁底仓回测事件，并显式限制每日交易手数。

    相比 build_locked_signals，本函数新增“交易预算”控制，适合解决
    每天触发次数过多、交易手数过高的问题。

    核心语义仍然和 sandbox.PositionBacktester 一致:
    - signal = 1  -> direction = 1，买入平空，形成多头暴露
    - signal = -1 -> direction = -1，卖出平多，形成空头暴露
    - hold_bars>0 时，到期生成反向 direction，恢复对锁

    手数控制参数:
    - max_daily_open_lots:
      每个交易日总共最多接受多少手“开暴露”信号。
      若设为 20，hold_bars 存在时，最终 regular 行通常最多约 40 行/日。
    - max_daily_open_lots_per_direction:
      每个交易日每个方向最多接受多少手。
    - max_active_open_lots:
      同一时刻最多允许多少手暴露尚未恢复。
    - max_active_open_lots_per_direction:
      同一时刻每个方向最多允许多少手暴露尚未恢复。
    - min_abs_value:
      如果 model_output 有 value 列，则只接受 abs(value) >= min_abs_value 的信号。
    - block_same_direction_reentry:
      同方向暴露未恢复前，不再接受同方向新信号。
    - block_opposite_direction_reentry:
      任一反方向暴露未恢复前，不接受当前方向新信号。

    注意:
    - 所有 cap 都只限制“开暴露”事件，restore_lock 不计入开仓预算。
    - 为避免未来函数，函数按时间顺序逐条接受信号，不做“日内最强信号排序”。
    """
    required_cols = {"code", "trade_time", "signal"}
    missing = required_cols - set(model_output.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if base_position <= 0:
        raise ValueError("base_position must be > 0")
    if lot_per_signal <= 0:
        raise ValueError("lot_per_signal must be > 0")
    if cooldown_bars < 0:
        raise ValueError("cooldown_bars must be >= 0")
    if hold_bars is not None and hold_bars <= 0:
        raise ValueError("hold_bars must be > 0 when provided")
    if entry_resampling_win is not None and entry_resampling_win <= 0:
        raise ValueError("entry_resampling_win must be > 0 when provided")

    cap_values = {
        "max_daily_open_lots": max_daily_open_lots,
        "max_daily_open_lots_per_direction": max_daily_open_lots_per_direction,
        "max_active_open_lots": max_active_open_lots,
        "max_active_open_lots_per_direction":
        max_active_open_lots_per_direction,
    }
    for name, value in cap_values.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be > 0 when provided")

    if min_abs_value is not None and min_abs_value < 0:
        raise ValueError("min_abs_value must be >= 0 when provided")

    # 复制输入，避免在调用方的 DataFrame 上留下临时列。
    df = model_output.copy()
    df["trade_time"] = pd.to_datetime(df["trade_time"])

    # date_col 用来对齐交易日定义。
    # 期货夜盘可能跨自然日，因此如果上游已有 trade_date，应优先使用它。
    if date_col is not None:
        if date_col not in df.columns:
            raise ValueError(
                f"date_col={date_col!r} not found in model_output")
        df["_signal_date"] = pd.to_datetime(df[date_col]).dt.normalize()
    else:
        df["_signal_date"] = df["trade_time"].dt.normalize()

    # sandbox.PositionBacktester 使用 date + min_time 定位成交价，
    # 因此这里显式生成 min_time，保持输出格式和原始回测器兼容。
    df["_min_time"] = df["trade_time"].dt.strftime("%H%M")
    df = df.sort_values(["code", "_signal_date",
                         "trade_time"]).reset_index(drop=True)

    records = []
    value_col = "value" if "value" in df.columns else None

    for (code, signal_date), group in df.groupby(["code", "_signal_date"],
                                                 sort=True):
        # 对锁底仓的第一性原理:
        # 每天先假设有 base_position 手多头底仓 + base_position 手空头底仓。
        # signal=1 时，买入平空，空头底仓减少，账户产生净多暴露。
        # signal=-1 时，卖出平多，多头底仓减少，账户产生净空暴露。
        # hold_bars 到期后，反向交易恢复原来的对锁状态。
        remaining_by_direction = {1: base_position, -1: base_position}

        # daily_open_lots_by_direction 控制“今天已经接受了多少手开暴露”。
        # 这个计数不会因为 restore_lock 而减少，因为它衡量的是日内交易预算。
        daily_open_lots_by_direction = {1: 0, -1: 0}

        # active_open_lots_by_direction 控制“当前还有多少手暴露尚未恢复”。
        # 这个计数会在 open_exposure 时增加，在 restore_lock 时减少。
        active_open_lots_by_direction = {1: 0, -1: 0}

        # cooldown 使用最近一次接受信号的位置，而不是最近一次出现信号的位置。
        # 这样被过滤掉的弱信号不会意外拉长冷却时间。
        last_accept_idx_by_direction = {1: None, -1: None}

        # 用 bar 索引调度未来的恢复对锁事件。
        # 例如 i 点开暴露、hold_bars=5，则在 i+5 点生成反向事件。
        restore_events_by_idx = {}

        signals = group["signal"].to_numpy()
        trade_times = group["trade_time"].to_numpy()
        min_times = group["_min_time"].to_numpy()
        minutes = group["trade_time"].dt.minute.to_numpy()
        values = (group[value_col].to_numpy(dtype=float, copy=False)
                  if value_col else None)

        for i, raw_signal in enumerate(signals):
            # 先处理到期恢复事件，再判断当前 bar 是否允许新开暴露。
            # 这样同一根 bar 上可以先释放额度，再根据新信号重新占用额度。
            restore_events = restore_events_by_idx.pop(i, None)
            if restore_events:
                for event in restore_events:
                    records.append(event["record"])
                    opened_direction = event["opened_direction"]
                    numbers = event["numbers"]
                    remaining_by_direction[opened_direction] += numbers
                    active_open_lots_by_direction[opened_direction] -= numbers

            # signal 只接受 -1/0/1。
            # 0 表示没有方向优势，不生成任何交易事件。
            signal = int(raw_signal)
            if signal == 0:
                continue
            if signal not in (-1, 1):
                raise ValueError(f"signal must be in {{-1,0,1}}, got {signal}")

            # 对齐 FactorEvaluate1 的 resampling_win。
            # 例如 period=5 时，只允许 09:30、09:35、09:40 这种 5 分钟栅格入场。
            # 注意这里不能预先抽样数据，否则 hold_bars=5 会变成 5 个抽样点，即 25 分钟。
            if entry_resampling_win and entry_resampling_win > 1:
                if int(minutes[i]) % int(entry_resampling_win) != 0:
                    continue

            # 如果要求固定持有期，则必须确认未来存在可恢复对锁的 bar。
            # 尾部不足 hold_bars 的信号会被丢弃，避免产生无法按期恢复的暴露。
            close_idx = None
            if hold_bars is not None:
                close_idx = i + hold_bars
                if close_idx >= len(group):
                    continue

            # 可选强度过滤。
            # signal 决定方向，value 只用于衡量当前方向的置信度/强度。
            if values is not None and min_abs_value is not None:
                if abs(float(values[i])) < float(min_abs_value):
                    continue

            # 同方向仍有暴露时不重复进入，可以显著降低连续同向信号导致的滚动交易。
            if block_same_direction_reentry:
                if active_open_lots_by_direction[signal] > 0:
                    continue

            # 反方向仍有暴露时也不进入，用于要求“同一时刻只表达一个方向观点”的场景。
            if block_opposite_direction_reentry:
                if active_open_lots_by_direction[-signal] > 0:
                    continue

            # cooldown 是更粗的频率控制。
            # 它不关心暴露是否已经恢复，只要求同方向两次接受信号之间至少间隔 N 根 bar。
            if cooldown_bars > 0:
                last_idx = last_accept_idx_by_direction[signal]
                if last_idx is not None and (i - last_idx) < cooldown_bars:
                    continue

            # cap_candidates 收集所有“本次最多能交易多少手”的约束。
            # 最终 numbers 取最小值，相当于同时满足底仓、日预算、活跃暴露预算等所有约束。
            cap_candidates = [
                int(lot_per_signal), remaining_by_direction[signal]
            ]

            if max_daily_open_lots is not None:
                daily_total = sum(daily_open_lots_by_direction.values())
                cap_candidates.append(int(max_daily_open_lots) - daily_total)

            if max_daily_open_lots_per_direction is not None:
                cap_candidates.append(
                    int(max_daily_open_lots_per_direction) -
                    daily_open_lots_by_direction[signal])

            if max_active_open_lots is not None:
                active_total = sum(active_open_lots_by_direction.values())
                cap_candidates.append(int(max_active_open_lots) - active_total)

            if max_active_open_lots_per_direction is not None:
                cap_candidates.append(
                    int(max_active_open_lots_per_direction) -
                    active_open_lots_by_direction[signal])

            numbers = min(cap_candidates)
            if numbers <= 0:
                continue

            # 接受当前信号:
            # 1. 消耗对应方向的底仓额度。
            # 2. 增加日内已开暴露手数。
            # 3. 增加当前未恢复暴露手数。
            # 4. 记录最近接受信号的位置，用于 cooldown。
            remaining_by_direction[signal] -= numbers
            daily_open_lots_by_direction[signal] += numbers
            active_open_lots_by_direction[signal] += numbers
            last_accept_idx_by_direction[signal] = i

            daily_open_total = sum(daily_open_lots_by_direction.values())
            active_open_total = sum(active_open_lots_by_direction.values())

            record = {
                "date":
                pd.Timestamp(signal_date),
                "min_time":
                str(min_times[i]),
                "code":
                code,
                "direction":
                signal,
                "numbers":
                int(numbers),
                "trade_time":
                pd.Timestamp(trade_times[i]),
                "remaining_same_side":
                int(remaining_by_direction[signal]),
                "daily_open_lots_total":
                int(daily_open_total),
                "daily_open_lots_side":
                int(daily_open_lots_by_direction[signal]),
                "active_open_lots_total":
                int(active_open_total),
                "active_open_lots_side":
                int(active_open_lots_by_direction[signal]),
                "signal_type":
                "open_exposure" if hold_bars is not None else "consume_lock",
            }
            if values is not None:
                record["source_value"] = float(values[i])
            records.append(record)

            if hold_bars is not None:
                # 到期恢复对锁:
                # 当前 signal=1 买入平空后，未来需要 direction=-1 卖出恢复空头底仓。
                # 当前 signal=-1 卖出平多后，未来需要 direction=1 买入恢复多头底仓。
                # restore_lock 不计入日内开暴露预算，因为它不是新观点，而是风险复位。
                restore_record = {
                    "date":
                    pd.Timestamp(signal_date),
                    "min_time":
                    str(min_times[close_idx]),
                    "code":
                    code,
                    "direction":
                    -signal,
                    "numbers":
                    int(numbers),
                    "trade_time":
                    pd.Timestamp(trade_times[close_idx]),
                    "remaining_same_side":
                    int(remaining_by_direction[signal]),
                    "daily_open_lots_total":
                    int(daily_open_total),
                    "daily_open_lots_side":
                    int(daily_open_lots_by_direction[signal]),
                    "active_open_lots_total":
                    int(active_open_total),
                    "active_open_lots_side":
                    int(active_open_lots_by_direction[signal]),
                    "signal_type":
                    "restore_lock",
                }
                if values is not None:
                    restore_record["source_value"] = float(values[i])
                restore_events_by_idx.setdefault(close_idx, []).append({
                    "opened_direction":
                    signal,
                    "numbers":
                    int(numbers),
                    "record":
                    restore_record,
                })

    if not records:
        columns = [
            "date", "min_time", "code", "direction", "numbers", "trade_time",
            "remaining_same_side", "daily_open_lots_total",
            "daily_open_lots_side", "active_open_lots_total",
            "active_open_lots_side", "signal_type"
        ]
        if value_col:
            columns.append("source_value")
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(records)
    result["_signal_type_order"] = result["signal_type"].map({
        "restore_lock": 0,
        "open_exposure": 1,
        "consume_lock": 1,
    }).fillna(9)
    result = result.sort_values(
        ["code", "date", "min_time",
         "_signal_type_order"]).reset_index(drop=True)
    return result.drop(columns=["_signal_type_order"])


# def build_position_signals(model_output: pd.DataFrame,
#                            hold_bars: int = 15,
#                            max_position: int = 10,
#                            lot_per_signal: int = 1,
#                            cooldown_bars: int = 0,
#                            prevent_same_direction_reentry: bool = True):
#     """
#     根据离散信号列 `signal` 构造开平仓指令。

#     输入要求:
#     - model_output 必须包含:
#       - code
#       - trade_time
#       - value   : 连续值，仅用于记录 source_value
#       - signal  : 离散信号，取值为 -1 / 0 / 1

#     逻辑:
#     - signal = 1  -> 开多
#     - signal = -1 -> 开空
#     - signal = 0  -> 不开仓
#     - 开仓后固定持有 hold_bars 根 bar
#     - 到期自动反向生成平仓信号
#     - 同时持仓总手数不超过 max_position
#     - 若 prevent_same_direction_reentry=True，则已有同方向持仓时不重复开仓
#     - 若 cooldown_bars > 0，则同方向开仓后需要等待 cooldown_bars 根 bar 才允许再次开仓
#     """
#     required_cols = {"code", "trade_time", "value", "signal"}
#     missing = required_cols - set(model_output.columns)
#     if missing:
#         raise ValueError(f"missing required columns: {missing}")

#     if hold_bars <= 0:
#         raise ValueError("hold_bars must be > 0")
#     if max_position <= 0:
#         raise ValueError("max_position must be > 0")
#     if lot_per_signal <= 0:
#         raise ValueError("lot_per_signal must be > 0")
#     if cooldown_bars < 0:
#         raise ValueError("cooldown_bars must be >= 0")

#     df = model_output.copy()
#     df["trade_time"] = pd.to_datetime(df["trade_time"])
#     df = df.sort_values(["code", "trade_time"]).reset_index(drop=True)

#     records = []
#     pair_id = 0

#     for code, group in df.groupby("code", sort=True):
#         group = group.sort_values("trade_time").reset_index(drop=True)

#         active_positions = []
#         active_lots = 0
#         last_open_idx_by_direction = {1: None, -1: None}

#         for i, row in group.iterrows():
#             current_time = row["trade_time"]

#             # 1. 先处理当前 bar 到期的平仓，这样同一根 bar 上的后续开仓判断
#             #    是基于“已经释放掉到期仓位”的真实状态。
#             remaining = []
#             for pos in active_positions:
#                 if pos["close_idx"] == i:
#                     records.append({
#                         "date": current_time.normalize(),
#                         "min_time": current_time.strftime("%H%M"),
#                         "code": code,
#                         "direction": -pos["direction"],
#                         "numbers": pos["numbers"],
#                         "signal_type": "close",
#                         "pair_id": pos["pair_id"],
#                         "source_value": pos["source_value"],
#                         "open_trade_time": pos["open_trade_time"],
#                         "close_trade_time": pos["close_trade_time"],
#                     })
#                     active_lots -= pos["numbers"]
#                 else:
#                     remaining.append(pos)
#             active_positions = remaining

#             # 2. 只使用离散 signal，不再基于 value 再做一次方向判断。
#             signal = int(row["signal"])
#             if signal == 0:
#                 continue
#             if signal not in (-1, 1):
#                 raise ValueError(f"signal must be in {{-1,0,1}}, got {signal}")

#             # 3. 若已有同方向持仓，则跳过，避免同方向信号连续出现时不断滚动加仓。
#             if prevent_same_direction_reentry:
#                 has_same_direction = any(pos["direction"] == signal
#                                          for pos in active_positions)
#                 if has_same_direction:
#                     continue

#             # 4. 同方向 cooldown。即使旧仓已经平掉，也要求等待若干 bar 后才能再次开同方向。
#             if cooldown_bars > 0:
#                 last_open_idx = last_open_idx_by_direction[signal]
#                 if last_open_idx is not None and (
#                         i - last_open_idx) < cooldown_bars:
#                     continue

#             # 5. 仓位上限控制。限制的是同时持有的总手数，而不是一天交易次数。
#             if active_lots + lot_per_signal > max_position:
#                 continue

#             # 6. 若剩余 bar 不足持有期，则不再开仓，避免生成无法按规则平掉的尾部仓位。
#             close_idx = i + hold_bars
#             if close_idx >= len(group):
#                 continue

#             open_direction = signal
#             close_time = group.iloc[close_idx]["trade_time"]
#             source_value = float(row["value"])

#             pair_id += 1
#             records.append({
#                 "date": current_time.normalize(),
#                 "min_time": current_time.strftime("%H%M"),
#                 "code": code,
#                 "direction": open_direction,
#                 "numbers": lot_per_signal,
#                 "signal_type": "open",
#                 "pair_id": pair_id,
#                 "source_value": source_value,
#                 "open_trade_time": current_time,
#                 "close_trade_time": close_time,
#             })

#             active_positions.append({
#                 "direction": open_direction,
#                 "numbers": lot_per_signal,
#                 "close_idx": close_idx,
#                 "pair_id": pair_id,
#                 "source_value": source_value,
#                 "open_trade_time": current_time,
#                 "close_trade_time": close_time,
#             })
#             active_lots += lot_per_signal
#             last_open_idx_by_direction[signal] = i

#     position_df = pd.DataFrame(records)
#     if position_df.empty:
#         return pd.DataFrame(columns=[
#             "date", "min_time", "code", "direction", "numbers", "pair_id",
#             "source_value", "open_trade_time", "close_trade_time"
#         ])

#     position_df["signal_trade_time"] = pd.to_datetime(
#         position_df["date"].astype(str) + " " +
#         position_df["min_time"].str.slice(0, 2) + ":" +
#         position_df["min_time"].str.slice(2, 4) + ":00")

#     # 同一时间戳下，先执行 close 再执行 open，和上面的生成逻辑保持一致。
#     position_df["signal_type_order"] = position_df["signal_type"].map({
#         "close":
#         0,
#         "open":
#         1,
#     }).fillna(9)

#     position_df = position_df.sort_values(
#         ["code", "signal_trade_time", "signal_type_order",
#          "pair_id"]).reset_index(drop=True)

#     position_df = position_df.drop(
#         columns=["signal_trade_time", "signal_type_order", "signal_type"])
#     return position_df


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
        group = group.reset_index(drop=True)
        n = len(group)
        trade_times = group["trade_time"].to_numpy()
        signals = group["signal"].to_numpy()
        values = group["value"].to_numpy(dtype=float, copy=False)

        close_events_by_idx = {}
        active_count_by_direction = {1: 0, -1: 0}
        active_lots = 0
        last_open_idx_by_direction = {1: None, -1: None}

        for i in range(n):
            current_time = pd.Timestamp(trade_times[i])

            # 1. 先处理当前 bar 到期的平仓，这样同一根 bar 上的后续开仓判断
            #    是基于“已经释放掉到期仓位”的真实状态。
            close_events = close_events_by_idx.pop(i, None)
            if close_events:
                for pos in close_events:
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
                    active_count_by_direction[
                        pos["direction"]] -= pos["numbers"]

            # 2. 只使用离散 signal，不再基于 value 再做一次方向判断。
            signal = int(signals[i])
            if signal == 0:
                continue
            if signal not in (-1, 1):
                raise ValueError(f"signal must be in {{-1,0,1}}, got {signal}")

            # 3. 若已有同方向持仓，则跳过，避免同方向信号连续出现时不断滚动加仓。
            if prevent_same_direction_reentry:
                if active_count_by_direction[signal] > 0:
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
            if close_idx >= n:
                continue

            open_direction = signal
            close_time = pd.Timestamp(trade_times[close_idx])
            source_value = float(values[i])

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

            pos = {
                "direction": open_direction,
                "numbers": lot_per_signal,
                "close_idx": close_idx,
                "pair_id": pair_id,
                "source_value": source_value,
                "open_trade_time": current_time,
                "close_trade_time": close_time,
            }
            close_events_by_idx.setdefault(close_idx, []).append(pos)
            active_lots += lot_per_signal
            active_count_by_direction[open_direction] += lot_per_signal
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


def load_market_data(instruments, begin_time, end_time, trading_sessions):
    market_data = fetch_research_data(instruments=instruments,
                                      begin_time=begin_time,
                                      end_time=end_time,
                                      adjusted_method=None)

    market_data = filter_trading_time(data=market_data,
                                      trading_sessions=trading_sessions)
    return market_data


period = 5


def load_backtest_results(method, instruments, task_id, param_id=None, sub_path=None):
    if isinstance(param_id, str):
        basic_path = os.path.join(base_path, method,
                                  instruments, 'temp', 'model', str(task_id),
                                  str(period), 'rl', 'backtest',
                                  'equal_weight', param_id)
    else:
        basic_path = os.path.join(base_path, method,
                                  instruments, 'temp', 'model', str(task_id),
                                  str(period), 'rl', 'backtest')


    basic_path = Path(basic_path)
    res = []
    if not basic_path.exists():
        print(f"Warning: Root path does not exist: {basic_path}")
        return []
    
    if sub_path:
        # **核心修改点**
        # a. 构建一个 glob 搜索模式。
        #    '**' 是一个通配符，代表“任意多层子目录”。
        #    所以这个模式的意思是：在 root_path 下的任何地方，找到匹配 sub_path 的目录，
        #    然后再在那个目录里找到 'daily_stats.feather' 文件。
        search_pattern = f"**/{sub_path}/**/daily_stats.feather"
        files_to_load = basic_path.glob(search_pattern)
    else:
        # 如果没有提供 sub_path，则递归搜索所有的 'daily_stats.feather'
        files_to_load = basic_path.rglob('daily_stats.feather')
    
    for feat_file in files_to_load:
        # 从文件路径中提取有意义的名称
        # 这个逻辑依然适用：取文件的上两级目录名作为标识
        #print(feat_file.parts)
        name = f"{feat_file.parts[-6]}_{feat_file.parts[-3]}_{feat_file.parts[-2]}"
        
        try:
            pnl_series = pd.read_feather(feat_file).set_index('date')['cumulative_pnl']
            pnl_series.name = name
            res.append(pnl_series)
        except Exception as e:
            print(f"Failed to load or process file {feat_file}: {e}")
            
    return res


def _direction_key(direction: int) -> str:
    return str(int(direction))


def _init_live_locked_context(base_position: int) -> dict:
    return {
        "bar_index": -1,
        "last_trade_time": None,
        "remaining_by_direction": {
            "1": int(base_position),
            "-1": int(base_position),
        },
        "daily_open_lots_by_direction": {
            "1": 0,
            "-1": 0,
        },
        "active_open_lots_by_direction": {
            "1": 0,
            "-1": 0,
        },
        "last_accept_idx_by_direction": {
            "1": None,
            "-1": None,
        },
        "restore_events_by_idx": {},
    }



def _live_state_context(state: dict, code: str, signal_date: pd.Timestamp,
                        base_position: int) -> dict:
    if state is None:
        state = {}
    state.setdefault("contexts", {})

    code_key = str(code)
    date_key = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    state["contexts"].setdefault(code_key, {})
    if date_key not in state["contexts"][code_key]:
        state["contexts"][code_key][date_key] = _init_live_locked_context(
            base_position)

    return state["contexts"][code_key][date_key]


def _empty_live_event_frame(has_value: bool = True) -> pd.DataFrame:
    columns = [
        "date", "min_time", "code", "direction", "numbers", "trade_time",
        "remaining_same_side", "daily_open_lots_total",
        "daily_open_lots_side", "active_open_lots_total",
        "active_open_lots_side", "signal_type"
    ]
    if has_value:
        columns.append("source_value")
    return pd.DataFrame(columns=columns)


def live_capped_locked_step(
        bar_signal,
        state: dict = None,
        base_position: int = 10,
        lot_per_signal: int = 1,
        cooldown_bars: int = 0,
        hold_bars: int = None,
        entry_resampling_win: int = None,
        date_col: str = "trade_date",
        max_daily_open_lots: int = None,
        max_daily_open_lots_per_direction: int = None,
        max_active_open_lots: int = None,
        max_active_open_lots_per_direction: int = None,
        min_abs_value: float = None,
        block_same_direction_reentry: bool = False,
        block_opposite_direction_reentry: bool = False,
        extend_same_direction: bool = True):
    """
    实盘逐 bar 版本的 capped locked 执行器。

    这个函数解决 build_capped_locked_signals 不能直接用于实盘的问题:
    - 回测函数一次性看到完整 DataFrame，可以用 i + hold_bars 找未来平仓点。
    - 实盘每分钟只看到当前 bar，因此必须把未来 restore_lock 事件写入 state。

    输入:
    - bar_signal 可以是一行 dict / pd.Series，也可以是多行 pd.DataFrame。
    - 必须包含 code、trade_time、signal。
    - 建议包含 trade_date，尤其期货夜盘必须用交易日而不是自然日。
    - 可选包含 value，用于 min_abs_value 过滤和 source_value 记录。

    输出:
    - events_df: 当前输入 bar 触发的交易事件，字段和 build_capped_locked_signals 对齐。
    - state: 更新后的执行状态。实盘应持久化这个 dict，重启后继续传回。

    实盘下单语义:
    - signal_type == open_exposure:
      direction=1  表示买入平空，形成多头暴露。
      direction=-1 表示卖出平多，形成空头暴露。
    - signal_type == restore_lock:
      direction 是 open_exposure 的反向，用于恢复对锁底仓。

    注意:
    - 本函数只处理执行状态，不生成 signal；signal 应由模型/信号函数提前算好。
    - 若临近收盘不允许新开暴露，应由调用方把 signal 置 0，或不要调用开仓逻辑。
    """
    required_cols = {"code", "trade_time", "signal"}
    if state is None:
        state = {"contexts": {}}

    if isinstance(bar_signal, pd.DataFrame):
        df = bar_signal.copy()
    elif isinstance(bar_signal, pd.Series):
        df = bar_signal.to_frame().T
    elif isinstance(bar_signal, dict):
        df = pd.DataFrame([bar_signal])
    else:
        raise TypeError(
            "bar_signal must be a dict, pd.Series, or pd.DataFrame")

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if base_position <= 0:
        raise ValueError("base_position must be > 0")
    if lot_per_signal <= 0:
        raise ValueError("lot_per_signal must be > 0")
    if cooldown_bars < 0:
        raise ValueError("cooldown_bars must be >= 0")
    if hold_bars is not None and hold_bars <= 0:
        raise ValueError("hold_bars must be > 0 when provided")
    if entry_resampling_win is not None and entry_resampling_win <= 0:
        raise ValueError("entry_resampling_win must be > 0 when provided")
    if min_abs_value is not None and min_abs_value < 0:
        raise ValueError("min_abs_value must be >= 0 when provided")

    cap_values = {
        "max_daily_open_lots": max_daily_open_lots,
        "max_daily_open_lots_per_direction": max_daily_open_lots_per_direction,
        "max_active_open_lots": max_active_open_lots,
        "max_active_open_lots_per_direction":
        max_active_open_lots_per_direction,
    }
    for name, value in cap_values.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be > 0 when provided")

    # ---------- 3. 交易日归属 ----------
    # 期货夜盘非常关键：trade_time 的自然日不一定等于交易日。
    # 如果输入有 trade_date，就用 trade_date 分组；否则才退化为自然日。
    pdb.set_trace()
    
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    if date_col is not None and date_col in df.columns:
        df["_signal_date"] = pd.to_datetime(df[date_col]).dt.normalize()
    else:
        df["_signal_date"] = df["trade_time"].dt.normalize()
    df = df.sort_values(["code", "_signal_date", "trade_time"])

    has_value = "value" in df.columns
    records = []

    for row in df.itertuples():
        code = row.code
        trade_time = pd.Timestamp(row["trade_time"])
        signal_date = pd.Timestamp(row["_signal_date"])
        signal = int(row["signal"])
        value = float(row["value"]) if has_value else None

        if signal not in (-1, 0, 1):
            raise ValueError(f"signal must be in {{-1,0,1}}, got {signal}")

        ctx = _live_state_context(state, code, signal_date, base_position)
        last_trade_time = ctx.get("last_trade_time")
        if last_trade_time is not None:
            if trade_time < pd.Timestamp(last_trade_time):
                raise ValueError(
                    f"trade_time must be non-decreasing for {code}: "
                    f"{trade_time} < {last_trade_time}")

        ctx["bar_index"] = int(ctx.get("bar_index", -1)) + 1
        ctx["last_trade_time"] = trade_time.isoformat()
        bar_idx = int(ctx["bar_index"])
        min_time = trade_time.strftime("%H%M")

        remaining = ctx["remaining_by_direction"]
        daily_open = ctx["daily_open_lots_by_direction"]
        active_open = ctx["active_open_lots_by_direction"]
        last_accept = ctx["last_accept_idx_by_direction"]
        restore_events = ctx["restore_events_by_idx"]

        is_entry_bar = True
        if entry_resampling_win and entry_resampling_win > 1:
            is_entry_bar = int(trade_time.minute) % int(
                entry_resampling_win) == 0

        due_key = str(bar_idx)
        due_events = restore_events.pop(due_key, [])
        used_signal_for_extension = False

        for event in due_events:
            opened_direction = int(event["opened_direction"])
            opened_key = _direction_key(opened_direction)
            numbers = int(event["numbers"])

            can_extend = (
                extend_same_direction and hold_bars is not None
                and signal == opened_direction and signal in (-1, 1)
                and is_entry_bar)
            if can_extend:
                next_due_idx = bar_idx + int(hold_bars)
                restore_events.setdefault(str(next_due_idx), []).append({
                    "opened_direction": opened_direction,
                    "numbers": numbers,
                    "source_value": value,
                })
                used_signal_for_extension = True
                continue

            # restore_lock 恢复对锁底仓，同时释放活跃暴露额度。
            remaining[opened_key] = int(remaining[opened_key]) + numbers
            active_open[opened_key] = max(
                0,
                int(active_open[opened_key]) - numbers)

            active_total = sum(int(x) for x in active_open.values())
            daily_total = sum(int(x) for x in daily_open.values())
            record = {
                "date": signal_date,
                "min_time": min_time,
                "code": code,
                "direction": -opened_direction,
                "numbers": numbers,
                "trade_time": trade_time,
                "remaining_same_side": int(remaining[opened_key]),
                "daily_open_lots_total": int(daily_total),
                "daily_open_lots_side": int(daily_open[opened_key]),
                "active_open_lots_total": int(active_total),
                "active_open_lots_side": int(active_open[opened_key]),
                "signal_type": "restore_lock",
            }
            if has_value:
                record["source_value"] = event.get("source_value", value)
            records.append(record)

        if used_signal_for_extension:
            continue

        if signal == 0:
            continue
        if not is_entry_bar:
            continue

        if has_value and min_abs_value is not None:
            if abs(value) < float(min_abs_value):
                continue

        signal_key = _direction_key(signal)
        opposite_key = _direction_key(-signal)

        if block_same_direction_reentry:
            if int(active_open[signal_key]) > 0:
                continue

        if block_opposite_direction_reentry:
            if int(active_open[opposite_key]) > 0:
                continue

        if cooldown_bars > 0:
            last_idx = last_accept[signal_key]
            if last_idx is not None and (bar_idx - int(last_idx)) < int(
                    cooldown_bars):
                continue

        cap_candidates = [int(lot_per_signal), int(remaining[signal_key])]

        if max_daily_open_lots is not None:
            daily_total = sum(int(x) for x in daily_open.values())
            cap_candidates.append(int(max_daily_open_lots) - daily_total)

        if max_daily_open_lots_per_direction is not None:
            cap_candidates.append(
                int(max_daily_open_lots_per_direction) -
                int(daily_open[signal_key]))

        if max_active_open_lots is not None:
            active_total = sum(int(x) for x in active_open.values())
            cap_candidates.append(int(max_active_open_lots) - active_total)

        if max_active_open_lots_per_direction is not None:
            cap_candidates.append(
                int(max_active_open_lots_per_direction) -
                int(active_open[signal_key]))

        numbers = min(cap_candidates)
        if numbers <= 0:
            continue

        remaining[signal_key] = int(remaining[signal_key]) - int(numbers)
        daily_open[signal_key] = int(daily_open[signal_key]) + int(numbers)
        active_open[signal_key] = int(active_open[signal_key]) + int(numbers)
        last_accept[signal_key] = int(bar_idx)

        daily_total = sum(int(x) for x in daily_open.values())
        active_total = sum(int(x) for x in active_open.values())
        record = {
            "date": signal_date,
            "min_time": min_time,
            "code": code,
            "direction": signal,
            "numbers": int(numbers),
            "trade_time": trade_time,
            "remaining_same_side": int(remaining[signal_key]),
            "daily_open_lots_total": int(daily_total),
            "daily_open_lots_side": int(daily_open[signal_key]),
            "active_open_lots_total": int(active_total),
            "active_open_lots_side": int(active_open[signal_key]),
            "signal_type":
            "open_exposure" if hold_bars is not None else "consume_lock",
        }
        if has_value:
            record["source_value"] = value
        records.append(record)

        if hold_bars is not None:
            due_idx = bar_idx + int(hold_bars)
            restore_events.setdefault(str(due_idx), []).append({
                "opened_direction": int(signal),
                "numbers": int(numbers),
                "source_value": value,
            })

    if not records:
        return _empty_live_event_frame(has_value=has_value), state

    result = pd.DataFrame(records)
    result["_signal_type_order"] = result["signal_type"].map({
        "restore_lock": 0,
        "open_exposure": 1,
        "consume_lock": 1,
    }).fillna(9)
    result = result.sort_values(
        ["code", "date", "min_time",
         "_signal_type_order"]).reset_index(drop=True)
    return result.drop(columns=["_signal_type_order"]), state
