import pdb
import os
import datetime as dt
import re
import pandas as pd
from joblib import Parallel, delayed
from ultron.tradingday import advanceDateByCalendar
from lib.data.load_data import fetch_algin_factors


FILENAME_TRADE_DATE_PATTERN = re.compile(r"_(\d{8})\.csv$")


def _previous_trading_date(trading_date: dt.date) -> dt.date:
    previous_day = advanceDateByCalendar("china.sse", trading_date, "-1b")
    if isinstance(previous_day, dt.datetime):
        return previous_day.date()
    return previous_day


def _infer_anchor_trading_date(
    csv_path: str,
    df: pd.DataFrame,
    day_session_start: dt.time,
    night_session_start: dt.time,
) -> dt.date:
    match = FILENAME_TRADE_DATE_PATTERN.search(str(csv_path))
    if match:
        return dt.datetime.strptime(match.group(1), "%Y%m%d").date()

    update_times = pd.to_datetime(df["UpdateTime"], format="%H:%M:%S")
    update_clock = update_times.dt.time
    day_mask = (update_clock >= day_session_start) & (
        update_clock < night_session_start)

    if day_mask.any():
        anchor_value = df.loc[day_mask, "TradingDay"].mode().iloc[0]
    else:
        anchor_value = df["TradingDay"].max()
    return dt.datetime.strptime(str(int(anchor_value)), "%Y%m%d").date()


def _prepare_tick_datetimes(
    csv_path: str,
    df: pd.DataFrame,
    night_session_start: dt.time,
    day_session_start: dt.time,
    overnight_session_end: dt.time,
) -> pd.Series:
    anchor_trading_date = _infer_anchor_trading_date(
        csv_path=csv_path,
        df=df,
        day_session_start=day_session_start,
        night_session_start=night_session_start,
    )
    previous_anchor_date = _previous_trading_date(anchor_trading_date)
    overnight_natural_date = previous_anchor_date + dt.timedelta(days=1)

    update_times = pd.to_datetime(df["UpdateTime"], format="%H:%M:%S")
    update_clock = update_times.dt.time

    night_mask = update_clock >= night_session_start
    overnight_mask = update_clock <= overnight_session_end

    anchor_ts = pd.Timestamp(anchor_trading_date)
    previous_anchor_ts = pd.Timestamp(previous_anchor_date)
    overnight_ts = pd.Timestamp(overnight_natural_date)

    # Normalize by trading-session rules instead of trusting raw TradingDay,
    # because different contracts/files may already have shifted TradingDay
    # differently. The file name date is treated as the target TradingDay T:
    # - evening segment -> previous trading day natural date
    # - overnight segment -> previous trading day's next calendar day
    # - day segment -> TradingDay natural date
    natural_dates = pd.Series(
        anchor_ts, index=df.index, dtype="datetime64[ns]")
    natural_dates.loc[night_mask] = previous_anchor_ts
    natural_dates.loc[overnight_mask] = overnight_ts

    millis = df["UpdateMillisec"].fillna(
        0).astype(int).astype(str).str.zfill(3)
    datetime_text = (
        natural_dates.dt.strftime("%Y-%m-%d")
        + " "
        + df["UpdateTime"]
        + "."
        + millis
    )
    return pd.to_datetime(datetime_text, format="%Y-%m-%d %H:%M:%S.%f")


def parallecl_tick_data(tick_file, main_file, code):
    tick_data = pd.read_csv(tick_file)
    tick_data["TickDatetime"] = _prepare_tick_datetimes(
        csv_path=tick_file,
        df=tick_data,
        night_session_start=dt.time(20, 0, 0),
        day_session_start=dt.time(9, 0, 0),
        overnight_session_end=dt.time(4, 0, 0)
    )
    tick_data = tick_data.sort_values(["TickDatetime", "UpdateMillisec"], kind="stable").reset_index(
        drop=True)
    tick_data['Code'] = code
    # '/workspace/data/fut_tick/7050707549_-/2025/202503/20250303/rb2505_20250303.csv' 被破坏
    print(main_file)
    tick_data.to_feather(main_file)


def process_tick_data(base_path,
                      begin_date,
                      end_date,
                      codes,
                      output_path):
    algin_factors = fetch_algin_factors(
        begin_date=begin_date,
        end_date=end_date,
        codes=codes,
        columns=['trade_date', 'code', 'symbol'])
    algin_factors = algin_factors.drop_duplicates(
        subset=['trade_date', 'code'], keep='last')

    file_res = []
    for row in algin_factors.itertuples():
        print(row)
        output_dirs = os.path.join(output_path, row.code)
        filename1 = os.path.join(output_dirs, "{0}.feather".format(
            row.trade_date.strftime('%Y%m%d')))
        if os.path.exists(filename1):
            continue

        os.makedirs(output_dirs, exist_ok=True)
        filename = os.path.join(base_path, row.trade_date.strftime('%Y'), row.trade_date.strftime('%Y%m'),
                                row.trade_date.strftime('%Y%m%d'), "{0}_{1}.csv".format(row.symbol, row.trade_date.strftime('%Y%m%d')))
        if not os.path.exists(filename):
            filename = os.path.join(base_path, row.trade_date.strftime('%Y'), row.trade_date.strftime('%Y%m'), row.trade_date.strftime('%Y%m'),
                                    row.trade_date.strftime('%Y%m%d'), "{0}_{1}.csv".format(row.symbol, row.trade_date.strftime('%Y%m%d')))
        file_res.append((filename, filename1, row.code))

    _ = Parallel(n_jobs=64, verbose=1)(delayed(parallecl_tick_data)(
        tick_file=file_group[0],
        main_file=file_group[1],
        code=file_group[2]) for file_group in file_res)

    # tick_data = pd.read_csv(filename)
    # tick_data["TickDatetime"] = _prepare_tick_datetimes(
    #     csv_path=filename,
    #     df=tick_data,
    #     night_session_start=dt.time(20, 0, 0),
    #     day_session_start=dt.time(9, 0, 0),
    #     overnight_session_end=dt.time(4, 0, 0)
    # )
    # tick_data = tick_data.sort_values(["TickDatetime", "UpdateMillisec"], kind="stable").reset_index(
    #     drop=True
    # )
    # tick_data['Code'] = row.code
    # os.makedirs(output_dirs, exist_ok=True)
    # # '/workspace/data/fut_tick/7050707549_-/2025/202503/20250303/rb2505_20250303.csv' 被破坏
    # tick_data.to_feather(filename1)
