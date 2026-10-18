"""按显式来源日及固定合约复权；不解释或覆盖主力角色。"""
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import re
from zoneinfo import ZoneInfo

from .base import HistoryError
from ..sources import ReferenceSourceFactory, ReferenceQuery, DataSourcePurpose


class ReferenceFactorAdjuster:
    def __init__(self, config, source_days, *, backend="dolphindb", factory=ReferenceSourceFactory,
                 timezone="Asia/Shanghai"):
        self.source_days = dict(source_days)
        if any(type(day) is not date or type(source) is not date or source >= day
                for day, source in self.source_days.items()):
            raise HistoryError("须显式映射交易日期到更早的因子来源日，不推断或跨日沿用")
        self.zone = ZoneInfo(timezone)
        self.config, self.backend, self.factory = config, backend, factory
        self.last_audit = []

    def __call__(self, bars):
        bars = tuple(bars)
        required = {}
        for bar in bars:
            if bar.adjusted_close is not None:
                continue
            day = datetime.fromtimestamp(bar.ts_event // 1000000000, self.zone).date()
            if day not in self.source_days:
                raise HistoryError(f"{day}未声明因子来源日，不能沿用其他日期因子")
            match = re.fullmatch(r"([A-Za-z]+)\d+\.[A-Z0-9]+", bar.instrument_id)
            if match is None or bar.close is None:
                raise HistoryError("历史复权须为固定真实合约且包含原始收盘价")
            symbol = bar.instrument_id.rsplit(".", 1)[0].upper()
            key = (day, bar.instrument_id)
            required[key] = (match[1].upper(), symbol, self.source_days[day])
        factors, audits = {}, []
        if required:
            source = self.factory.create_for(DataSourcePurpose.LIVE_REFERENCE, self.backend, self.config)
            try:
                source.open()
                for key, (product, symbol, source_day) in sorted(required.items()):
                    batch = source.adjustment_factors(ReferenceQuery(products=(product,), symbols=(symbol,),
                        start_date=source_day, end_date=source_day))
                    if len(batch.rows) != 1:
                        raise HistoryError(f"{source_day}/{symbol}须有唯一累计因子，实际{len(batch.rows)}行")
                    row = batch.rows[0]
                    if (row["symbol"] != symbol or row["code"] != product or row["trade_date"] != source_day):
                        raise HistoryError("因子日期/合约/品种与请求不一致，禁止换合约或回退")
                    factor = Decimal(str(row["pcr_cumfactor"]))
                    if not factor.is_finite() or factor <= 0:
                        raise HistoryError("历史累计因子须为正且有限")
                    available = int(datetime.combine(source_day + timedelta(days=1), time(), self.zone).timestamp()) * 10**9
                    policy = "declared_source_day_end"
                    if row.get("available_ns") is not None:
                        published = Decimal(str(row["available_ns"]))
                        if not published.is_finite() or published < 0 or published != published.to_integral_value():
                            raise HistoryError("因子available_ns须为非负整数纳秒")
                        available = max(available, int(published))
                        policy = "explicit_and_source_day_end"
                    factors[key] = (factor, available)
                    audits.append({"trading_day": str(key[0]), "instrument": key[1],
                        "source_day": str(source_day), "factor": str(factor),
                        "available_ns": available, "availability_policy": policy,
                        "source": batch.source, "fingerprint": batch.fingerprint,
                        "version_evidence": "current_read_not_historical_revision_asof"})
            finally:
                source.close()
        adjusted = []
        for bar in bars:
            if bar.adjusted_close is not None:
                adjusted.append(bar)
                continue
            day = datetime.fromtimestamp(bar.ts_event // 1000000000, self.zone).date()
            factor, available = factors[(day, bar.instrument_id)]
            if available > bar.ts_event:
                raise HistoryError("因子在该分钟尚不可用，禁止提前复权")
            adjusted.append(replace(bar, adjusted_close=bar.close * factor, cumulative_factor=factor))
        self.last_audit = audits
        return tuple(adjusted)
