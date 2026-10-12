"""在线期权链和同月期货条款快照；来源连接与策略选约均由外层装配。"""
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import hashlib
import time
from types import MappingProxyType

from .metadata import parse_options_basic, parse_cffex_futures
from .sources.base import ReferenceQuery


@dataclass(frozen=True)
class OptionReferenceSnapshot:
    options: object
    futures: object
    fingerprint: str
    observed_at_ns: int


class LiveOptionReferences:
    """固定估值交易日的资料会话；刷新发现条款或合约集合变化时要求重新装配。

    基础资料的date/listDate是上市日期，不伪装成每日发布日；有available_ns时
    必须校验。缺少发布版本时仅记录内容指纹与实际读取时刻。
    """
    refresh_on_event = False

    def __init__(self, source, *, option_product, future_product, index_code, valuation_day,
                 refresh_seconds=5, max_observation_seconds=30, max_refresh_seconds=30,
                 clock_ns=None):
        self.source = source
        self.option_product, self.future_product, self.index_code = option_product, future_product, index_code
        self.valuation_day = valuation_day
        if type(valuation_day) is not date:
            raise ValueError("期权资料须声明估值交易日")
        self.clock_ns = clock_ns or time.time_ns
        values = (refresh_seconds, max_observation_seconds, max_refresh_seconds)
        if any(not Decimal(str(v)).is_finite() for v in values) or refresh_seconds < 0 or min(values[1:]) <= 0:
            raise ValueError("参考刷新参数无效")
        self.refresh_ns = int(refresh_seconds * 1e9)
        self.max_observation_ns = int(max_observation_seconds * 1e9)
        self.max_refresh_ns = int(max_refresh_seconds * 1e9)
        self._snapshot = None
        self._last_attempt = None
        self.failure = None
        self.refresh(force=True)

    def _read(self):
        return (self.source.options_basic(ReferenceQuery(products=(self.option_product,), active_on=self.valuation_day)),
            self.source.futures_basic(ReferenceQuery(products=(self.future_product,), active_on=self.valuation_day)))

    @staticmethod
    def _fingerprint(batches):
        return hashlib.sha256("|".join(batch.fingerprint for batch in batches).encode()).hexdigest()

    def refresh(self, *, force=False):
        now = self.clock_ns()
        if not force and self._last_attempt is not None and 0 <= now - self._last_attempt < self.refresh_ns:
            return self.snapshot(now)
        self._last_attempt = now
        self.failure = "期权资料刷新未完成"
        first, second = self._read(), self._read()
        fingerprint = self._fingerprint(first)
        if fingerprint != self._fingerprint(second):
            raise RuntimeError("读取期间期权或期货条款变化")
        observed = self.clock_ns()
        if not 0 <= observed - now <= self.max_refresh_ns:
            raise RuntimeError("期权资料读取超时或时钟回退")
        frames = []
        for batch in second:
            frame = batch.frame()
            for row in batch.rows:
                published = row.get("available_ns")
                if published is not None:
                    value = Decimal(str(published))
                    if not value.is_finite() or value != value.to_integral_value() or not 0 <= value <= observed:
                        raise ValueError("期权资料发布时间无效或尚未发布")
            # 可见性已经校验；公共静态条款解析只接收条款字段。
            frames.append(frame.drop(columns=[c for c in ("available_ns", "source_version") if c in frame]))
        options = parse_options_basic(frames[0], self.option_product, self.index_code,
            kinds=("C",), require_month_dates=False, require_currency=True, source=second[0].source)
        futures = parse_cffex_futures(frames[1], self.future_product, require_execution=True, source=second[1].source)
        options = {key: value for key, value in options.items()
            if value.listed <= self.valuation_day <= value.last_trade
            and self.future_product + value.month[2:] in futures}
        if not options:
            raise ValueError("没有具备已上市同月对冲期货的认购期权")
        used = {self.future_product + value.month[2:] for value in options.values()}
        futures = {key: MappingProxyType(dict(value)) for key, value in futures.items() if key in used}
        if self._snapshot is not None and fingerprint != self._snapshot.fingerprint:
            raise RuntimeError("期权参考资料版本变化，停止并重新装配；旧持仓腿不会被删除")
        self._snapshot = OptionReferenceSnapshot(MappingProxyType(options), MappingProxyType(futures), fingerprint, observed)
        self.failure = None
        return self._snapshot

    def snapshot(self, now_ns=None):
        now = self.clock_ns() if now_ns is None else now_ns
        if self.failure or self._snapshot is None or not 0 <= now - self._snapshot.observed_at_ns <= self.max_observation_ns:
            raise RuntimeError(self.failure or "期权参考资料观测过期")
        return self._snapshot

    def manifest(self):
        # Diagnostic access must work after expiry/failure and source.close().
        # It cannot open the strategy gate: snapshot() retains strict checks.
        snap, now = self._snapshot, self.clock_ns()
        age = None if snap is None else now - snap.observed_at_ns
        ready = not self.failure and age is not None and 0 <= age <= self.max_observation_ns
        if snap is None:
            return {"valuation_day": str(self.valuation_day), "ready": False,
                "failure": self.failure or "期权参考资料未取得", "source": type(self.source).__name__}
        return {"valuation_day": str(self.valuation_day), "fingerprint": snap.fingerprint,
            "observed_at_ns": snap.observed_at_ns, "source": type(self.source).__name__,
            "ready": bool(ready), "failure": None if ready else self.failure or "期权参考资料观测过期或时钟回退",
            "observation_age_seconds": age / 1e9,
            "evidence": "content_hash_and_observation_not_publication_version",
            "options": len(snap.options), "futures": len(snap.futures)}
