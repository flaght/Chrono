import datetime as dt
from dataclasses import dataclass, asdict

import pandas as pd


RB_CONTRACT_MULTIPLIER = 10
DROP_FIRST_PARTIAL_BAR = True


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


class MinuteBarBuilder:
    def __init__(self, multiplier: int, drop_first_partial_bar: bool = True) -> None:
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
            self.current_bar["high"] = max(self.current_bar["high"], tick.last_price)
            self.current_bar["low"] = min(self.current_bar["low"], tick.last_price)
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
            "start_turnover": self.prev_tick.turnover if self.prev_tick else None,
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


def to_tick(row: pd.Series) -> TickData:
    trading_day = str(row["TradingDay"])
    date_str = f"{trading_day[:4]}-{trading_day[4:6]}-{trading_day[6:]}"
    tick_dt = dt.datetime.strptime(
        f"{date_str} {row['UpdateTime']}.{int(row['UpdateMillisec']):03d}",
        "%Y-%m-%d %H:%M:%S.%f",
    )
    symbol = row["InstrumentID"]
    return TickData(
        symbol=symbol,
        vt_symbol=f"{symbol}.SHFE",
        exchange="SHFE",
        datetime=tick_dt,
        last_price=float(row["LastPrice"]),
        volume=float(row["Volume"]),
        turnover=float(row["Turnover"]),
        open_interest=float(row["OpenInterest"]),
    )


def aggregate_minute_bars(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values(
        ["TradingDay", "UpdateTime", "UpdateMillisec"],
        kind="stable",
    ).reset_index(drop=True)

    builder = MinuteBarBuilder(
        multiplier=RB_CONTRACT_MULTIPLIER,
        drop_first_partial_bar=DROP_FIRST_PARTIAL_BAR,
    )

    for _, row in df.iterrows():
        builder.update_tick(to_tick(row))

    builder.flush()
    return pd.DataFrame(builder.result_bars)


if __name__ == "__main__":
    csv_path = "rb2610_20260520.csv"
    final_df = aggregate_minute_bars(csv_path)

    print("最终合成的 Bar 数据：")
    print(
        final_df[
            [
                "datetime",
                "vt_symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
                "vwap",
            ]
        ].head(10)
    )
