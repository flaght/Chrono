import datetime, itertools
from collections import namedtuple
from dotenv import load_dotenv

load_dotenv()

from lib.rl012.sandbox import PositionBacktester
from ultron.tradingday import *
from chaosmind.timing.sirius0001.workflow import WorkFlow
from lib.uvx import load_sirius_params
from lib.attr001.ftd001 import *


def build_position_signals(model_output: pd.DataFrame,
                           hold_bars: int = 15,
                           confidence_threshold: float = 0.8,
                           max_position: int = 10,
                           lot_per_signal: int = 1):

    required_cols = {"code", "trade_time", "value"}
    missing = required_cols - set(model_output.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if hold_bars <= 0:
        raise ValueError("hold_bars must be > 0")
    if max_position <= 0:
        raise ValueError("max_position must be > 0")
    if lot_per_signal <= 0:
        raise ValueError("lot_per_signal must be > 0")

    df = model_output.copy()
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df = df.sort_values(["code", "trade_time"]).reset_index(drop=True)

    records = []
    pair_id = 0

    for code, group in df.groupby("code", sort=True):
        group = group.sort_values("trade_time").reset_index(drop=True)
        active_positions = []
        active_lots = 0

        for i, row in group.iterrows():
            current_time = row["trade_time"]

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

            value = float(row["value"])
            if abs(value) < confidence_threshold:
                continue

            if active_lots + lot_per_signal > max_position:
                continue

            close_idx = i + hold_bars
            if close_idx >= len(group):
                continue

            open_direction = 1 if value > 0 else -1
            close_time = group.iloc[close_idx]["trade_time"]

            pair_id += 1
            records.append({
                "date": current_time.normalize(),
                "min_time": current_time.strftime("%H%M"),
                "code": code,
                "direction": open_direction,
                "numbers": lot_per_signal,
                "signal_type": "open",
                "pair_id": pair_id,
                "source_value": value,
                "open_trade_time": current_time,
                "close_trade_time": close_time,
            })

            active_positions.append({
                "direction": open_direction,
                "numbers": lot_per_signal,
                "close_idx": close_idx,
                "pair_id": pair_id,
                "source_value": value,
                "open_trade_time": current_time,
                "close_trade_time": close_time,
            })
            active_lots += lot_per_signal

    position_df = pd.DataFrame(records)
    if position_df.empty:
        return pd.DataFrame(columns=[
            "date", "min_time", "code", "direction", "numbers", "signal_type",
            "pair_id", "source_value", "open_trade_time", "close_trade_time"
        ])

    position_df["signal_trade_time"] = pd.to_datetime(
        position_df["date"].astype(str) + " " +
        position_df["min_time"].str.slice(0, 2) + ":" +
        position_df["min_time"].str.slice(2, 4) + ":00")

    position_df["signal_type_order"] = position_df["signal_type"].map({
        "open":
        0,
        "close":
        1,
    }).fillna(9)

    position_df = position_df.sort_values(
        ["code", "signal_trade_time", "signal_type_order",
         "pair_id"]).reset_index(drop=True)

    position_df = position_df.drop(
        columns=["signal_trade_time", "signal_type_order", "signal_type"])
    return position_df


def run_source(fetch_market_func, mongo_client, trading_sessions,
               factors_infos, params, category, instruments, begin_time,
               end_time, start_time):
    market_data = fetch_market_func(instruments=instruments,
                                    begin_time=start_time,
                                    end_time=end_time,
                                    adjusted_method=None)

    market_data = filter_trading_time(data=market_data,
                                      trading_sessions=trading_sessions)

    netout_data = fetch_netout(
        category=category,
        code=INSTRUMENTS_CODES[instruments],
        begin_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
        table_name="netout_series",
        mongo_client=mongo_client)

    pdb.set_trace()
    position_data = build_position_signals(model_output=netout_data,
                                           hold_bars=5,
                                           confidence_threshold=0.94,
                                           max_position=1,
                                           lot_per_signal=1)
    pb = PositionBacktester(market_data=market_data, contract_multiplier=10)
    trade_records, daily_stats = pb.run(position_df=position_data, code='RB')
    pdb.set_trace()
    print('-->')


def start1(instruments, task_id):
    category = ['bench', 'research', 'trader']
    mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
    trading_sessions = (("21:00", "23:00"), ("09:00", "10:15"),
                        ("10:30", "11:30"), ("13:30", "15:00"))

    begin_time = datetime.datetime(2026, 5, 7)
    end_time = datetime.datetime(2026, 5, 26)

    start_time = advanceDateByCalendar('china.sse', begin_time, '-1b')
    end_time1 = advanceDateByCalendar('china.sse', end_time, '1b')

    factors_infos, params = load_sirius_params(
        code=INSTRUMENTS_CODES[instruments], task_id=task_id)

    source_configs = [
        ("bench", fetch_bench_data),
        ("research", fetch_research_data),
        ("trader", fetch_trader_data),
    ]

    for category, fetch_market_func in source_configs:
        run_source(fetch_market_func=fetch_market_func,
                   mongo_client=mongo_client,
                   trading_sessions=trading_sessions,
                   factors_infos=factors_infos,
                   params=params,
                   category=category,
                   instruments=instruments,
                   begin_time=begin_time,
                   end_time=end_time,
                   start_time=start_time)


if __name__ == '__main__':
    start1(instruments='rbb', task_id='1029921127239410')
