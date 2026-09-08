"""
FileDataLoader - 从本地 Parquet 加载数据并转换为 Nautilus 类型。

数据组织：
    data/IC.parquet, IF.parquet, ...  ← 期货
    data/idx.parquet                  ← 指数
    data/opt_1min.parquet             ← 期权1分钟
    data/opt_5min.parquet             ← 期权5分钟

加载逻辑：
    load_bars("IC1803", "M1")        → data/IC.parquet
    load_bars("SH000905", "M1")      → data/idx.parquet
    load_bars("MO2509-P-5200", "M5") → data/opt_5min.parquet
"""
import os
from decimal import Decimal

import pandas as pd

from bomber.model.data import Bar, BarType, BarSpecification, BarAggregation
from bomber.model.data import QuoteTick, TradeTick
from bomber.model.identifiers import InstrumentId, Symbol, Venue
from bomber.model.objects import Price, Quantity
from bomber.model.enums import AggressorSide, PriceType
from bomber.model.instruments.base import Instrument
from bomber.persistence.wranglers import BarDataWrangler

from .enums import PERIOD_MAP

# 带这些前缀的代码去 idx.parquet 查，其余按前2字母去 {root}.parquet
_IDX_PREFIXES = ("SH", "SZ", "000", "399", "899")
# 期权合约识别（含 "-" 或第3字符为数字的代码如 510050P... → opt_*min.parquet）
_IS_OPTION = lambda c: "-" in c or (len(c) > 2 and c[2:3].isdigit() and any(k in c for k in ("P", "C")) and len(c) > 6)


class FileDataLoader:
    """从 Parquet/CSV 加载行情数据"""

    def __init__(self, data_dir: str = "", venue: str = "CFFEX"):
        self._data_dir = data_dir
        self._venue = Venue(venue)

    def load_bars(self, instrument_id: str, period: str = "M1",
                  filepath: str = None, instrument=None,
                  period_start: str = None, period_end: str = None) -> list:
        if filepath is None:
            bars = self._load_from_parquet(instrument_id, period, instrument,
                                          period_start, period_end)
            # parquet 为空是该合约在日期范围内无数据，不回退 CSV
            return bars
        return self._load_from_csv(filepath, instrument_id, period, instrument,
                                  period_start, period_end)

    def _is_index(self, code: str) -> bool:
        return any(code.startswith(p) for p in _IDX_PREFIXES)

    def _load_from_parquet(self, instrument_id: str, period: str,
                           instrument=None,
                           period_start: str = None, period_end: str = None) -> list:
        # 确定 parquet 文件名
        if self._is_index(instrument_id):
            pq_name = "idx.parquet"
        elif _IS_OPTION(instrument_id):
            pq_name = f"opt_{period.lower()}.parquet"
        else:
            pq_name = f"{instrument_id[:2]}.parquet"

        pq_path = os.path.join(self._data_dir, pq_name)
        if not os.path.exists(pq_path):
            return []

        try:
            df = pd.read_parquet(pq_path, filters=[("code", "==", instrument_id)])
        except Exception:
            df = pd.read_parquet(pq_path)
            df = df[df["code"] == instrument_id]

        if df.empty:
            return []

        df = df.set_index("timestamp").sort_index()

        # 按时间范围过滤（Timestamp 比较，兼容 tz-aware index）
        if period_start or period_end:
            ts_start = pd.Timestamp(period_start).tz_localize('Asia/Shanghai') if period_start else None
            ts_end = pd.Timestamp(period_end).tz_localize('Asia/Shanghai') if period_end else None
            if ts_start:
                df = df[df.index >= ts_start]
            if ts_end:
                df = df[df.index < ts_end]

        # 兼容旧数据：若 timestamp 无时区，标注为 Asia/Shanghai（CST=UTC+8）
        if df.index.tz is None:
            df.index = df.index.tz_localize('Asia/Shanghai')
        return self._df_to_bars(df, instrument_id, period, instrument)

    def _df_to_bars(self, df, instrument_id, period, instrument=None) -> list:
        if df is None or df.empty:
            return []
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        bad = ((df["high"] < df["open"]) |
               (df["low"] > df["open"]) |
               (df["high"] < df["low"]))
        if bad.any():
            df.loc[bad, "high"] = df.loc[bad, ["open", "high"]].max(axis=1)
            df.loc[bad, "low"] = df.loc[bad, ["open", "low"]].min(axis=1)

        df = df[["open", "high", "low", "close", "volume"]]

        inst_id = InstrumentId(Symbol(instrument_id), self._venue)
        agg_name, step = PERIOD_MAP.get(period, ("MINUTE", 1))
        agg_map = {
            "SECOND": BarAggregation.SECOND, "MINUTE": BarAggregation.MINUTE,
            "HOUR": BarAggregation.HOUR, "DAY": BarAggregation.DAY,
            "WEEK": BarAggregation.WEEK,
        }
        bar_spec = BarSpecification(step, agg_map.get(agg_name, BarAggregation.MINUTE),
                                    PriceType.LAST)
        bar_type = BarType(inst_id, bar_spec)

        if instrument is not None:
            wrangler = BarDataWrangler(bar_type, instrument)
            bars = wrangler.process(df)
        else:
            precision = instrument.price_precision if instrument else 2
            bars = []
            for idx, row in df.iterrows():
                ts_ns = int(idx.timestamp() * 1e9)
                bars.append(Bar(
                    bar_type=bar_type,
                    open=Price(float(row["open"]), precision),
                    high=Price(float(row["high"]), precision),
                    low=Price(float(row["low"]), precision),
                    close=Price(float(row["close"]), precision),
                    volume=Quantity(float(row.get("volume", 0)), 0),
                    ts_event=ts_ns, ts_init=ts_ns,
                ))

        print(f"[FileDataLoader] Loaded {len(bars)} bars for {instrument_id} from parquet")
        return bars

    def _load_from_csv(self, filepath, instrument_id, period, instrument=None,
                       period_start: str = None, period_end: str = None) -> list:
        if not os.path.exists(filepath):
            print(f"[FileDataLoader] File not found: {filepath}")
            return []

        df = pd.read_csv(filepath)
        col_map = {
            "timeStamp": "timestamp", "ActTime": "timestamp", "time": "timestamp",
            "openPrice": "open", "highPrice": "high",
            "lowPrice": "low", "closePrice": "close",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        for col in ["timestamp", "open", "high", "low", "close"]:
            if col not in df.columns:
                print(f"[FileDataLoader] Missing column: {col}")
                return []

        if df["timestamp"].dtype == object:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif df["timestamp"].iloc[0] > 1e15:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")

        df = df.set_index("timestamp").sort_index()

        # 按时间范围过滤（Timestamp 比较，兼容 tz-aware index）
        if period_start or period_end:
            ts_start = pd.Timestamp(period_start).tz_localize('Asia/Shanghai') if period_start else None
            ts_end = pd.Timestamp(period_end).tz_localize('Asia/Shanghai') if period_end else None
            if ts_start:
                df = df[df.index >= ts_start]
            if ts_end:
                df = df[df.index < ts_end]

        # 兼容旧数据：若 timestamp 无时区，标注为 Asia/Shanghai（CST=UTC+8）
        if df.index.tz is None:
            df.index = df.index.tz_localize('Asia/Shanghai')
        if "volume" not in df.columns:
            df["volume"] = 0
        return self._df_to_bars(df, instrument_id, period, instrument)

    def load_ticks(self, instrument_id: str, filepath: str = None) -> tuple:
        if filepath is None:
            filepath = os.path.join(self._data_dir, "ic", "ticks", f"{instrument_id}.csv")
        if not os.path.exists(filepath):
            return [], []

        df = pd.read_csv(filepath)
        col_map = {
            "timeStamp": "timestamp", "time": "timestamp",
            "last_price": "lastPrice", "bid_price": "bidPrice",
            "ask_price": "askPrice", "bid_volume": "bidVolume",
            "ask_volume": "askVolume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "timestamp" in df.columns:
            if df["timestamp"].dtype == object:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            elif df["timestamp"].iloc[0] > 1e15:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")
            df = df.set_index("timestamp")
        df = df.sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize('Asia/Shanghai')

        inst_id = InstrumentId(Symbol(instrument_id), self._venue)
        quotes, trades = [], []
        for idx, row in df.iterrows():
            ts_ns = int(idx.timestamp() * 1e9)
            bp, ap = float(row.get("bidPrice", 0)), float(row.get("askPrice", 0))
            if bp > 0 and ap > 0:
                quotes.append(QuoteTick(
                    instrument_id=inst_id,
                    bid_price=Price(bp, 2), ask_price=Price(ap, 2),
                    bid_size=Quantity(float(row.get("bidVolume", 0)), 0),
                    ask_size=Quantity(float(row.get("askVolume", 0)), 0),
                    ts_event=ts_ns, ts_init=ts_ns,
                ))
            lp = float(row.get("lastPrice", 0))
            v = float(row.get("volume", 0))
            if lp > 0 and v > 0:
                trades.append(TradeTick(
                    instrument_id=inst_id,
                    price=Price(lp, 2), size=Quantity(v, 0),
                    aggressor_side=AggressorSide.NO_AGGRESSOR,
                    trade_id=str(ts_ns), ts_event=ts_ns, ts_init=ts_ns,
                ))
        print(f"[FileDataLoader] Loaded {len(quotes)} quotes + {len(trades)} trades")
        return quotes, trades
