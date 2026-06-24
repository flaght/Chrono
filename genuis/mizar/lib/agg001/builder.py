import pdb
import datetime as dt
import re
import pandas as pd
from dataclasses import dataclass, asdict

from ultron.tradingday import advanceDateByCalendar


FILENAME_TRADE_DATE_PATTERN = re.compile(r"_(\d{8})\.csv$")


@dataclass
class TickData:
    symbol: str
    vt_symbol: str
    exchange: str
    datetime: dt.datetime
    last_price: float
    volume: float
    turnover: float
    open_interest: float


@dataclass
class BarData:
    vt_symbol: str
    symbol: str
    exchange: str
    open: float
    high: float
    low: float
    close: float
    date: str
    time: str
    datetime: str
    volume: float
    value: float
    open_interest: float
    vwap: float


class BarBuilder:

    def __init__(self,
                 multiplier: int,
                 drop_first_partial_bar: bool = True) -> None:
        self.multiplier = multiplier
        self.drop_first_partial_bar = drop_first_partial_bar
        self.current_minute = None
        self.current_bar = None
        self.prev_tick = None
        self.has_emitted_bar = False
        self.result_bars = []

    def update_tick(self, tick: TickData) -> None:
        minute_bucket = tick.datetime.replace(second=0, microsecond=0)

        if self.current_minute is None:
            self.current_minute = minute_bucket
            self.current_bar = self._new_bar(tick)
            self.prev_tick = tick
            return

        if minute_bucket != self.current_minute:
            self._flush_current_bar()
            self.current_minute = minute_bucket
            self.current_bar = self._new_bar(tick)
        else:
            self.current_bar["high"] = max(self.current_bar["high"],
                                           tick.last_price)
            self.current_bar["low"] = min(self.current_bar["low"],
                                          tick.last_price)
            self.current_bar["close"] = tick.last_price
            self.current_bar["end_volume"] = tick.volume
            self.current_bar["end_turnover"] = tick.turnover
            self.current_bar["open_interest"] = tick.open_interest

        self.prev_tick = tick

    def flush(self) -> None:
        self._flush_current_bar()

    def _new_bar(self, tick: TickData) -> dict:
        return {
            "vt_symbol": tick.vt_symbol,
            "symbol": tick.symbol,
            "exchange": tick.exchange,
            "minute": tick.datetime.replace(second=0, microsecond=0),
            "open": tick.last_price,
            "high": tick.last_price,
            "low": tick.last_price,
            "close": tick.last_price,
            "start_volume": self.prev_tick.volume if self.prev_tick else None,
            "end_volume": tick.volume,
            "start_turnover":
            self.prev_tick.turnover if self.prev_tick else None,
            "end_turnover": tick.turnover,
            "open_interest": tick.open_interest,
        }

    def _flush_current_bar(self) -> None:
        if not self.current_bar:
            return

        start_volume = self.current_bar["start_volume"]
        start_turnover = self.current_bar["start_turnover"]

        if start_volume is None or start_turnover is None:
            if self.drop_first_partial_bar:
                self.current_bar = None
                return
            volume = 0.0
            value = 0.0
        else:
            volume = max(0.0, self.current_bar["end_volume"] - start_volume)
            value = max(0.0, self.current_bar["end_turnover"] - start_turnover)

        minute_dt = self.current_bar["minute"]
        close_price = self.current_bar["close"]
        vwap = value / volume / self.multiplier if volume > 0 else close_price

        bar = BarData(
            vt_symbol=self.current_bar["vt_symbol"],
            symbol=self.current_bar["symbol"],
            exchange=self.current_bar["exchange"],
            open=self.current_bar["open"],
            high=self.current_bar["high"],
            low=self.current_bar["low"],
            close=close_price,
            date=minute_dt.strftime("%Y-%m-%d"),
            time=minute_dt.strftime("%H:%M:%S"),
            datetime=minute_dt.strftime("%Y-%m-%d %H:%M:%S"),
            volume=volume,
            value=value,
            open_interest=self.current_bar["open_interest"],
            vwap=vwap,
        )
        self.result_bars.append(asdict(bar))
        self.has_emitted_bar = True
        self.current_bar = None


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
    day_mask = (update_clock >= day_session_start) & (update_clock < night_session_start)

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
    natural_dates = pd.Series(anchor_ts, index=df.index, dtype="datetime64[ns]")
    natural_dates.loc[night_mask] = previous_anchor_ts
    natural_dates.loc[overnight_mask] = overnight_ts

    millis = df["UpdateMillisec"].fillna(0).astype(int).astype(str).str.zfill(3)
    datetime_text = (
        natural_dates.dt.strftime("%Y-%m-%d")
        + " "
        + df["UpdateTime"]
        + "."
        + millis
    )
    return pd.to_datetime(datetime_text, format="%Y-%m-%d %H:%M:%S.%f")


def to_tick(row: pd.Series,
            night_session_start,
            day_session_start,
            overnight_session_end,
            exchange) -> TickData:
    symbol = row.InstrumentID
    return TickData(
        symbol=symbol,
        vt_symbol=f"{symbol}.{exchange}",
        exchange=exchange,
        datetime=row.tick_datetime,
        last_price=float(row.LastPrice),
        volume=float(row.Volume),
        turnover=float(row.Turnover),
        open_interest=float(row.OpenInterest),
    )


def minute_bars(csv_path: str,
                multiplier: str,
                night_session_start: str,
                exchange: str,
                day_session_start: str = "09:00:00",
                overnight_session_end: str = "04:00:00",
                drop_first=True) -> pd.DataFrame:
    try:
        #csv_path = "/workspace/data/fut_tick/7050707549_-/2026/202606/20260601/SA609_20260601.csv"
        #csv_path = "/workspace/data/fut_tick/7050707549_-/2026/202606/20260601/ag2702_20260601.csv"
        #csv_path = "/workspace/data/fut_tick/7050707549_-/2026/202606/20260602/rb2610_20260602.csv"
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"[ERROR] 遭遇空文件，跳过读取: {csv_path}")
        return pd.DataFrame() # 返回空表，防止程序崩溃
        
    except Exception as e:
        print(f"[ERROR] 读取文件 {csv_path} 时发生未知错误: {e}")
        return pd.DataFrame()
    
    df = df.sort_values(
        ["TradingDay", "UpdateTime", "UpdateMillisec"],
        kind="stable",
    ).reset_index(drop=True)

    builder = BarBuilder(
        multiplier=multiplier,
        drop_first_partial_bar=drop_first,
    )
    if isinstance(night_session_start, str):
        night_session_start = dt.datetime.strptime(
            night_session_start, "%H:%M:%S").time()
    if isinstance(day_session_start, str):
        day_session_start = dt.datetime.strptime(
            day_session_start, "%H:%M:%S").time()
    if isinstance(overnight_session_end, str):
        overnight_session_end = dt.datetime.strptime(
            overnight_session_end, "%H:%M:%S").time()

    df["tick_datetime"] = _prepare_tick_datetimes(
        csv_path=csv_path,
        df=df,
        night_session_start=night_session_start,
        day_session_start=day_session_start,
        overnight_session_end=overnight_session_end,
    )
    df = df.sort_values(["tick_datetime", "UpdateMillisec"], kind="stable").reset_index(
        drop=True
    )

    for row in df.itertuples():
        builder.update_tick(
            to_tick(row=row,
                    night_session_start=night_session_start,
                    day_session_start=day_session_start,
                    overnight_session_end=overnight_session_end,
                    exchange=exchange))

    builder.flush()
    result_df = pd.DataFrame(builder.result_bars)
    result_df = result_df.sort_values(
        by='datetime') if not result_df.empty else result_df
    return result_df
