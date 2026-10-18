import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from kdutils.data import *
from config.contract import INSTRUMENTS_CODES

from lib.rl012.sandbox import PositionBacktester


# def build_dummy_position_df(market_data, code='RB', numbers=1):
#     data = market_data.copy()
#     data["trade_time"] = pd.to_datetime(data["trade_time"])
#     data["date"] = data["trade_time"].dt.normalize()
#     data["min_time"] = data["trade_time"].dt.strftime("%H%M")

#     # 只取一个品种
#     data = data[data["code"] == code].copy()

#     # 挑几根典型时间做测试信号
#     test_times = {"0900", "0945", "1030", "1100", "1330", "1450"}
#     data = data[data["min_time"].isin(test_times)].copy()

#     # 同一天内交替多空
#     data = data.sort_values(["date", "min_time"])
#     data["direction"] = [1 if i % 2 == 0 else -1 for i in range(len(data))]
#     data["numbers"] = numbers
#     data["Code"] = data["code"]

#     position_df = data[["date", "min_time", "code", "direction",
#                         "numbers"]].reset_index(drop=True)
#     return position_df

def build_dummy_position_df(
    market_data,
    code="RB",
    numbers=1,
    holding_minutes=15,
    reopen_gap_minutes=1,
    first_entry_time="0900",
    seed=42,
):
    """
    生成随机的成对信号：
    - t: 开仓信号
    - t+holding_minutes: 平仓信号（反向）
    - t+holding_minutes+reopen_gap_minutes: 下一笔可重新开仓

    direction:
    - 1  : 开多 / 平空
    - -1 : 开空 / 平多
    """
    rng = np.random.default_rng(seed)

    data = market_data.copy()
    data["trade_time"] = pd.to_datetime(data["trade_time"])
    data["date"] = data["trade_time"].dt.normalize()
    data["min_time"] = data["trade_time"].dt.strftime("%H%M")

    # 只取一个品种
    data = data[data["code"] == code].copy()
    data = data.sort_values("trade_time").reset_index(drop=True)

    available = set(zip(data["date"], data["trade_time"]))

    records = []

    for date, group in data.groupby("date"):
        group = group.sort_values("trade_time").reset_index(drop=True)

        start_row = group[group["min_time"] == first_entry_time]
        if start_row.empty:
            continue

        current_entry_time = start_row.iloc[0]["trade_time"]

        while True:
            close_signal_time = current_entry_time + pd.Timedelta(minutes=holding_minutes)
            next_entry_time = close_signal_time + pd.Timedelta(minutes=reopen_gap_minutes)

            if (date, current_entry_time) not in available:
                break
            if (date, close_signal_time) not in available:
                break

            # 随机决定本轮是开多还是开空
            open_direction = int(rng.choice([1, -1]))
            close_direction = -open_direction

            # 开仓信号
            records.append({
                "date": date,
                "min_time": current_entry_time.strftime("%H%M"),
                "code": code,
                "direction": open_direction,
                "numbers": numbers,
            })

            # 15分钟后平仓信号
            records.append({
                "date": date,
                "min_time": close_signal_time.strftime("%H%M"),
                "code": code,
                "direction": close_direction,
                "numbers": numbers,
            })

            if (date, next_entry_time) not in available:
                break

            current_entry_time = next_entry_time

    position_df = pd.DataFrame(records)
    if position_df.empty:
        return position_df

    position_df = position_df.sort_values(["date", "min_time"]).reset_index(drop=True)
    return position_df

def run(instruments, adjusted_method=None):
    pdb.set_trace()
    begin_time = datetime.datetime(2026, 5, 21)
    end_time = datetime.datetime(2026, 5, 26)

    market_data = fetch_local_market1(base_path=os.environ['BAR_FUT_DIRS'],
                                      begin_date=begin_time,
                                      end_date=end_time,
                                      codes=[INSTRUMENTS_CODES[instruments]],
                                      method=adjusted_method,
                                      keep_symbol=True)
    pdb.set_trace()
    position_df = build_dummy_position_df(market_data, code="RB", numbers=1)
    pb = PositionBacktester(market_data=market_data, contract_multiplier=10)
    trade_records, daily_stats = pb.run(position_df=position_df, code='RB')
    metrics2 = pb.metrics()
    
    


run(instruments='rbb')
