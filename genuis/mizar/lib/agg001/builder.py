import pdb
import datetime as dt
import pandas as pd
from dataclasses import dataclass, asdict

from ultron.tradingday import advanceDateByCalendar


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


def to_tick(row: pd.Series,
            night_session_start,
            day_session_start,
            overnight_session_end,
            exchange) -> TickData:
    trading_day = str(row.TradingDay)
    trading_date = dt.datetime.strptime(trading_day, "%Y%m%d").date()
    update_time = dt.datetime.strptime(row.UpdateTime, "%H:%M:%S").time()
    previous_trading_date = _previous_trading_date(trading_date)

    # Use real natural timestamps for Chinese futures:
    # 1. 21:00~23:59 belongs to the previous trading day's evening
    # 2. 00:00~overnight_session_end belongs to the next natural day of that evening
    # 3. day session (for example 08:59/09:00 and later) belongs to TradingDay
    if update_time >= night_session_start:
        natural_date = previous_trading_date
    elif update_time <= overnight_session_end:
        natural_date = previous_trading_date + dt.timedelta(days=1)
    else:
        natural_date = trading_date

    tick_dt = dt.datetime.strptime(
        f"{natural_date:%Y-%m-%d} {row.UpdateTime}.{int(row.UpdateMillisec):03d}",
        "%Y-%m-%d %H:%M:%S.%f",
    )
    symbol = row.InstrumentID
    return TickData(
        symbol=symbol,
        vt_symbol=f"{symbol}.{exchange}",
        exchange=exchange,
        datetime=tick_dt,
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
