"""
FileDataLoader - 从本地 Parquet 加载数据并转换为 Nautilus 类型。

数据组织：
    data/IC.parquet, IF.parquet, ...  ← 期货
    data/idx.parquet                  ← 指数1分钟
    data/idx_5min.parquet             ← 指数5分钟
    data/opt_1min.parquet             ← 期权1分钟
    data/opt_5min.parquet             ← 期权5分钟

加载逻辑：
    load_bars("IC1803", "M1")        → data/IC.parquet
    load_bars("SH000905", "M1")      → data/idx.parquet
    load_bars("SH000905", "M5")      → data/idx_5min.parquet
    load_bars("MO2509-P-5200", "M5") → data/opt_5min.parquet
"""
import os
import re
import numpy as np
from decimal import Decimal

import pandas as pd

from bomber.model.data import Bar, BarType, BarSpecification, BarAggregation
from bomber.model.data import QuoteTick, TradeTick
from bomber.model.identifiers import InstrumentId, Symbol, Venue
from bomber.model.objects import Price, Quantity
from bomber.model.enums import AggressorSide, PriceType
from bomber.model.instruments.base import Instrument
from bomber.persistence.wranglers import BarDataWrangler

from bomber_adapter.enums import PERIOD_MAP
from bomber_adapter.instrument_info import is_option, is_index


_PERIOD_FILE_SUFFIX = {
    "M1": "", "M5": "_5min", "M15": "_15min",
    "M30": "_30min", "H1": "_1hour", "D1": "_daily",
}


def resolve_pq_name(instrument_id: str, period: str) -> str:
    """
    根据合约和周期确定 parquet 文件路径（新结构）

    新结构: market_data/{data_type}/{period}/{product}.parquet
    - data_type: futures/index/options
    - period: 1min/5min/15min/30min/1hour/daily
    - product: 品种代码（如 IC, idx, IO, OP）

    期权数据已按品种拆分：
    - IO（沪深300股指期权）→ IO.parquet
    - OP（50ETF期权）→ OP.parquet
    """
    # 确定数据类型和基础文件名
    if is_index(instrument_id):
        data_type = "index"
        base_name = "idx"
    elif is_option(instrument_id):
        data_type = "options"
        # 期权按品种拆分：提取品种代码（IO、OP 等）
        m = re.match(r'^([A-Za-z]+)', instrument_id)
        base_name = m.group(1) if m else instrument_id[:2]
    else:
        data_type = "futures"
        m = re.match(r'^([A-Za-z]+)', instrument_id)
        base_name = m.group(1) if m else instrument_id[:2]

    # 确定周期目录（支持两种格式：M1/M5 或 1min/5min）
    period_upper = period.upper()
    period_dir = {
        "M1": "1min", "M5": "5min", "M15": "15min",
        "M30": "30min", "H1": "1hour", "D1": "daily",
        "1MIN": "1min", "5MIN": "5min", "15MIN": "15min",
        "30MIN": "30min", "1HOUR": "1hour", "DAILY": "daily"
    }.get(period_upper, "1min")

    # 构建完整路径
    return f"market_data/{data_type}/{period_dir}/{base_name}.parquet"


class FileDataLoader:
    """从 Parquet/CSV 加载行情数据"""

    def __init__(
        self,
        data_dir: str = "",
        venue: str = "CFFEX",
        bar_timestamp: str = "end",
    ):
        if bar_timestamp not in {"start", "end"}:
            raise ValueError("bar_timestamp must be 'start' or 'end'")
        env_data_dir = os.environ.get("BOMBER_DATA_DIR") or os.environ.get("BT_DATA_DIR")
        self._data_dir = data_dir or env_data_dir or "./data"
        self._venue = Venue(venue)
        self._bar_timestamp = bar_timestamp
        # Sidecar 字典：存储 Bar 的额外字段（Nautilus Bar 不支持动态属性）
        # Key: (instrument_id, timestamp_ns)
        # Value: dict with keys like 'turnover', 'turnoverAccumulate', etc.
        self._bar_extra_fields = {}
        # 优化：parquet 文件缓存，避免重复 IO
        # Key: parquet 文件路径
        # Value: 加载的 DataFrame
        self._parquet_cache = {}

    def get_bar_extra_fields(self):
        """获取 Bar 额外字段 sidecar 字典"""
        return self._bar_extra_fields

    def preload_product(self, product: str, period: str = "M1"):
        """
        预加载整个品种的 parquet 文件到内存缓存。

        参数:
            product: 品种代码（如 'IC', 'IF', 'idx'）
            period: K线周期（如 'M1', 'M5'）
        """
        pq_name = resolve_pq_name(product, period)
        pq_path = os.path.join(self._data_dir, pq_name)

        if pq_path in self._parquet_cache:
            print(f"[DataLoader] {product} {period} 已在缓存中")
            return

        if not os.path.exists(pq_path):
            print(f"[DataLoader] 文件不存在: {pq_path}")
            return

        print(f"[DataLoader] 预加载 {product} {period} ...")
        try:
            df = pd.read_parquet(pq_path)
            df = df.set_index("timestamp").sort_index()
            self._parquet_cache[pq_path] = df

            # 对于大文件（>100万行），预构建按合约索引的字典，加速后续单合约加载
            if len(df) > 1_000_000 and "code" in df.columns:
                print(f"[DataLoader] 构建合约索引（{len(df):,} 条数据）...")
                contract_index = {}
                for code, group in df.groupby("code"):
                    contract_index[code] = group
                # 存储索引（使用特殊键）
                self._parquet_cache[f"{pq_path}::__contract_index__"] = contract_index
                print(f"[DataLoader] 已索引 {len(contract_index)} 个合约")

            print(f"[DataLoader] 已加载 {len(df)} 条数据到缓存")
        except Exception as e:
            print(f"[DataLoader] 预加载失败: {e}")

    def preload_instruments(self, inst_periods: dict):
        """
        预加载多个合约所需的 parquet 文件到内存缓存。

        参数:
            inst_periods: {inst_id: period} 合约和周期的映射
        """
        from collections import defaultdict

        # 按 parquet 文件分组
        groups = defaultdict(set)  # pq_name -> set of periods
        for inst_id, period in inst_periods.items():
            pq_name = self._resolve_pq_name(inst_id, period)
            groups[pq_name].add(period)

        # 预加载每个文件
        total_loaded = 0
        for pq_name, periods in groups.items():
            pq_path = os.path.join(self._data_dir, pq_name)
            if pq_path in self._parquet_cache:
                continue

            if not os.path.exists(pq_path):
                continue

            try:
                df = pd.read_parquet(pq_path)
                df = df.set_index("timestamp").sort_index()

                # 缓存整个文件（如果文件不太大）
                if len(df) < 10_000_000:  # 1000万行以内才缓存
                    self._parquet_cache[pq_path] = df
                    total_loaded += len(df)

                # 对于大文件（>100万行），预构建按合约索引的字典，加速后续单合约加载
                if len(df) > 1_000_000 and "code" in df.columns:
                    contract_index_key = f"{pq_path}::__contract_index__"
                    if contract_index_key not in self._parquet_cache:
                        print(f"[DataLoader] 构建合约索引 {pq_name}（{len(df):,} 条数据）...")
                        contract_index = {}
                        for code, group in df.groupby("code"):
                            contract_index[code] = group
                        self._parquet_cache[contract_index_key] = contract_index
                        print(f"[DataLoader] 已索引 {len(contract_index)} 个合约")
            except Exception as e:
                print(f"[DataLoader] 预加载失败 {pq_name}: {e}")

        if total_loaded > 0:
            print(f"[DataLoader] 预加载完成：{len(groups)} 个文件，{total_loaded:,} 条数据")

    @staticmethod
    def _period_delta(period: str) -> pd.Timedelta:
        agg_name, step = PERIOD_MAP.get(period.upper(), ("MINUTE", 1))
        units = {
            "SECOND": "s", "MINUTE": "min", "HOUR": "h",
            "DAY": "D", "WEEK": "W",
        }
        return pd.to_timedelta(step, unit=units[agg_name])

    @staticmethod
    def _shanghai_timestamp(value):
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("Asia/Shanghai")
        return timestamp.tz_convert("Asia/Shanghai")

    def _normalize_bar_index(self, df, period):
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        if df.index.has_duplicates:
            duplicates = df.index[df.index.duplicated()].unique()[:3]
            print(f"[DataLoader] ⚠ 重复时间戳 {list(duplicates)}，保留最后一条")
            df = df[~df.index.duplicated(keep="last")]
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("Asia/Shanghai")
        if self._bar_timestamp == "start":
            df = df.copy()
            df.index = df.index + self._period_delta(period)
        return df

    def load_bars(self, instrument_id: str, period: str = "M1",
                  filepath: str = None, instrument=None,
                  period_start: str = None, period_end: str = None) -> list:
        if filepath is None:
            return self._load_from_parquet(instrument_id, period, instrument,
                                          period_start, period_end)
        if filepath.endswith('.parquet'):
            return self._load_from_parquet(instrument_id, period, instrument,
                                          period_start, period_end, pq_override=filepath)
        return self._load_from_csv(filepath, instrument_id, period, instrument,
                                  period_start, period_end)

    def load_bars_batch(self, inst_periods: dict, instruments: dict = None,
                        period_start: str = None, period_end: str = None) -> dict:
        """
        批量加载 K 线：按 parquet 文件分组，每个文件只读一次，内存中按合约切分。

        参数:
            inst_periods: {inst_id: period}
            instruments: {inst_id: instrument}（可选）
            period_start / period_end: 时间过滤

        返回:
            {inst_id: [bars]}
        """
        from collections import defaultdict

        groups = defaultdict(list)  # pq_name -> [(inst_id, period)]
        for inst_id, period in inst_periods.items():
            pq_name = self._resolve_pq_name(inst_id, period)
            groups[pq_name].append((inst_id, period))

        result = {}
        for pq_name, items in groups.items():
            pq_path = os.path.join(self._data_dir, pq_name)
            insts = [i for i, _ in items]
            periods = {i: p for i, p in items}

            if not os.path.exists(pq_path):
                # 主文件缺失：逐合约走完整逻辑（含 fallback 聚合）
                for inst_id in insts:
                    bars = self.load_bars(
                        inst_id, periods[inst_id],
                        instrument=(instruments or {}).get(inst_id),
                        period_start=period_start, period_end=period_end)
                    if bars:
                        result[inst_id] = bars
                continue

            # 优化：使用缓存或加载到缓存
            # 优先检查合约索引（大文件优化）
            contract_index_key = f"{pq_path}::__contract_index__"
            use_contract_index = contract_index_key in self._parquet_cache

            if use_contract_index:
                # 使用合约索引，直接获取每个合约的数据
                contract_index = self._parquet_cache[contract_index_key]
                print(f"[DataLoader] 使用合约索引加载 {pq_name} ({len(insts)} 个合约)")

                for inst_id in insts:
                    if inst_id not in contract_index:
                        continue
                    df_i = contract_index[inst_id]
                    if df_i.empty:
                        continue
                    period = periods[inst_id]

                    # 按时间范围过滤
                    if period_start or period_end:
                        ts_start = self._shanghai_timestamp(period_start)
                        ts_end = self._shanghai_timestamp(period_end)
                        if ts_start:
                            df_i = df_i[df_i.index >= ts_start]
                        if ts_end:
                            df_i = df_i[df_i.index < ts_end]

                    df_i = self._normalize_bar_index(df_i.copy(), period)
                    bars = self._df_to_bars(
                        df_i, inst_id, period,
                        instrument=(instruments or {}).get(inst_id))
                    if bars:
                        result[inst_id] = bars
                continue

            # 没有合约索引，使用原来的逻辑
            if pq_path in self._parquet_cache:
                df = self._parquet_cache[pq_path]
                print(f"[DataLoader] 从缓存加载 {pq_name} ({len(insts)} 个合约)")
            else:
                try:
                    df = pd.read_parquet(pq_path, filters=[("code", "in", insts)])
                except Exception:
                    df = pd.read_parquet(pq_path)
                    df = df[df["code"].isin(insts)]

                df = df.set_index("timestamp").sort_index()

                # 缓存整个文件（如果文件不太大）
                if len(df) < 10_000_000:  # 1000万行以内才缓存
                    self._parquet_cache[pq_path] = df
                    print(f"[DataLoader] 已缓存 {pq_name} ({len(df)} 条数据)")

            if df.empty:
                continue

            # 按时间范围过滤（Timestamp 比较，兼容 tz-aware index）
            if period_start or period_end:
                ts_start = self._shanghai_timestamp(period_start)
                ts_end = self._shanghai_timestamp(period_end)
                if ts_start:
                    df = df[df.index >= ts_start]
                if ts_end:
                    df = df[df.index < ts_end]

            for inst_id in insts:
                period = periods[inst_id]
                df_i = df[df["code"] == inst_id]
                if df_i.empty:
                    continue
                df_i = self._normalize_bar_index(df_i.copy(), period)
                bars = self._df_to_bars(
                    df_i, inst_id, period,
                    instrument=(instruments or {}).get(inst_id))
                if bars:
                    result[inst_id] = bars
        return result

    def _load_from_parquet(self, instrument_id: str, period: str,
                           instrument=None,
                           period_start: str = None, period_end: str = None,
                           pq_override: str = None) -> list:
        # 确定 parquet 文件名（所有周期都已提前落地）
        pq_name = self._resolve_pq_name(instrument_id, period)

        if pq_override:
            pq_path = pq_override
        else:
            pq_path = os.path.join(self._data_dir, pq_name)

        # 文件不存在时回退到更细周期聚合
        if not os.path.exists(pq_path):
            fallback = self._resolve_fallback(instrument_id, period)
            if fallback and os.path.exists(os.path.join(self._data_dir, fallback)):
                pq_path = os.path.join(self._data_dir, fallback)
                need_aggregate = True
            else:
                return []
        else:
            need_aggregate = False

        # 优化：使用缓存或加载到缓存
        # 优先检查合约索引（大文件优化）
        contract_index_key = f"{pq_path}::__contract_index__"
        if contract_index_key in self._parquet_cache:
            contract_index = self._parquet_cache[contract_index_key]
            if instrument_id in contract_index:
                df = contract_index[instrument_id]
            else:
                return []
        elif pq_path in self._parquet_cache:
            df_full = self._parquet_cache[pq_path]
            # 从缓存中过滤特定合约
            df = df_full[df_full["code"] == instrument_id]
        else:
            try:
                df_full = pd.read_parquet(pq_path)
                df_full = df_full.set_index("timestamp").sort_index()
                # 缓存整个文件（如果文件不太大）
                if len(df_full) < 10_000_000:  # 1000万行以内才缓存
                    self._parquet_cache[pq_path] = df_full
            except Exception:
                df_full = pd.read_parquet(pq_path)
                df_full = df_full.set_index("timestamp").sort_index()

            # 从缓存中过滤特定合约
            df = df_full[df_full["code"] == instrument_id]

        if df.empty:
            return []

        # df 已继承 df_full 的 timestamp 索引，只需确保排序
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # 如果需要聚合（回退到更细周期文件）
        if need_aggregate:
            df = self._aggregate_bars(df, period)
            if df.empty:
                return []

        df = self._normalize_bar_index(df, period)

        # 按时间范围过滤（Timestamp 比较，兼容 tz-aware index）
        if period_start or period_end:
            ts_start = self._shanghai_timestamp(period_start)
            ts_end = self._shanghai_timestamp(period_end)
            if ts_start:
                df = df[df.index >= ts_start]
            if ts_end:
                df = df[df.index < ts_end]

        return self._df_to_bars(df, instrument_id, period, instrument)

    # ============================================================
    # 文件名解析（统一走模块级 resolve_pq_name）
    # ============================================================
    def _resolve_pq_name(self, instrument_id: str, period: str) -> str:
        """根据合约和周期确定 parquet 文件名"""
        return resolve_pq_name(instrument_id, period)

    def _resolve_fallback(self, instrument_id: str, period: str) -> str:
        """当目标周期文件不存在时，返回更细周期的文件名用于聚合"""
        # 优先级：M1 > M5 > M15 > M30 > H1 > D1
        fallback_order = ["M1", "M5", "M15", "M30", "H1", "D1"]
        period_upper = period.upper()
        try:
            idx = fallback_order.index(period_upper)
        except ValueError:
            return None
        # 尝试更细的周期
        for finer in fallback_order[:idx]:
            fname = self._resolve_pq_name(instrument_id, finer)
            if os.path.exists(os.path.join(self._data_dir, fname)):
                return fname
        return None

    def _aggregate_bars(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """
        将 M1 bar 聚合成更高周期（M5, M15, H1, D1 等）。

        输入 df 的 index 是 timestamp（tz-aware Asia/Shanghai）。
        返回聚合后的 DataFrame，index 为聚合后 bar 的结束时间。
        """
        if df.empty:
            return df

        # 解析周期
        period_upper = period.upper()
        agg_map = {
            "M5": "5min", "M15": "15min", "M30": "30min",
            "H1": "1h", "D1": "D", "W1": "W",
        }
        rule = agg_map.get(period_upper)
        if rule is None:
            return df  # 无法聚合，原样返回

        # 按周期 resample
        # open: 第一根, high: max, low: min, close: 最后一根, volume: sum
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        # 只聚合存在的列
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

        # 日线聚合：按 trade_date 列分组（对齐交易所口径，夜盘归次日）
        if rule == "D" and "trade_date" in df.columns:
            day_agg = {k: v for k, v in agg_dict.items() if k != "trade_date"}
            result = df.groupby("trade_date").agg(day_agg).dropna(subset=["open"])
            result.index = pd.to_datetime(result.index)
            if result.index.tz is None:
                result.index = result.index.tz_localize("Asia/Shanghai")
        else:
            result = df.resample(rule).agg(agg_dict).dropna(subset=["open"])

        if result.empty:
            return result

        # 确保时区一致
        if result.index.tz is None and df.index.tz is not None:
            result.index = result.index.tz_localize(df.index.tz)

        return result

    def _df_to_bars(self, df, instrument_id, period, instrument=None) -> list:
        if df is None or df.empty:
            return []
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        bad = ((df["high"] < df["open"]) |
               (df["high"] < df["close"]) |
               (df["low"] > df["open"]) |
               (df["low"] > df["close"]) |
               (df["high"] < df["low"]))
        if bad.any():
            df.loc[bad, "high"] = df.loc[bad, ["open", "high", "close"]].max(axis=1)
            df.loc[bad, "low"] = df.loc[bad, ["open", "low", "close"]].min(axis=1)

        # 提取额外字段到 sidecar 字典（在过滤列之前）
        extra_cols = [
            "turnover", "turnover_accumulate", "volume_accumulate", "open_interest_accumulate",
            "sectional_low", "sectional_high", "sectional_open"
        ]
        existing_extra_cols = [c for c in extra_cols if c in df.columns]
        if existing_extra_cols:
            # 性能优化：批量提取为 numpy 数组，比 iterrows 快 10-50x
            extra_values = df[existing_extra_cols].fillna(0.0).to_numpy(dtype=float)
            timestamps_ns = df.index.astype('int64').to_numpy()
            for i in range(len(df)):
                key = (instrument_id, int(timestamps_ns[i]))
                # 只存储非零值，减少内存占用；bridge.py 使用 extra.get(key, 0.0) 访问
                extra = {col: float(extra_values[i, j])
                         for j, col in enumerate(existing_extra_cols)
                         if extra_values[i, j] != 0.0}
                if extra:
                    self._bar_extra_fields[key] = extra

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

        if instrument is not None and not is_option(instrument_id):
            wrangler = BarDataWrangler(bar_type, instrument)
            bars = wrangler.process(df)
        else:
            precision = instrument.price_precision if instrument else 2
            # 性能优化：预提取为 numpy 数组，避免 iterrows 逐行访问 DataFrame 的开销
            opens = df["open"].to_numpy(dtype=float)
            highs = df["high"].to_numpy(dtype=float)
            lows = df["low"].to_numpy(dtype=float)
            closes = df["close"].to_numpy(dtype=float)
            volumes = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else np.zeros(len(df))
            timestamps_ns = df.index.astype('int64').to_numpy()

            bars = []
            for i in range(len(df)):
                bars.append(Bar(
                    bar_type=bar_type,
                    open=Price(float(opens[i]), precision),
                    high=Price(float(highs[i]), precision),
                    low=Price(float(lows[i]), precision),
                    close=Price(float(closes[i]), precision),
                    volume=Quantity(float(volumes[i]), 0),
                    ts_event=int(timestamps_ns[i]), ts_init=int(timestamps_ns[i]),
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
        df = self._normalize_bar_index(df, period)

        # 按时间范围过滤（Timestamp 比较，兼容 tz-aware index）
        if period_start or period_end:
            ts_start = self._shanghai_timestamp(period_start)
            ts_end = self._shanghai_timestamp(period_end)
            if ts_start:
                df = df[df.index >= ts_start]
            if ts_end:
                df = df[df.index < ts_end]

        # 兼容旧数据：若 timestamp 无时区，标注为 Asia/Shanghai（CST=UTC+8）
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
