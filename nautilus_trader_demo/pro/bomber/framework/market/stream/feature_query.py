"""将配置声明的真实OHLCV投影为Bar；查询来源和数据集不绑定具体品种。"""
import math
from decimal import Decimal
from threading import RLock

from bomber.framework.datahub.features import Alignment, FeatureQuery, FeatureRequirement, FeatureUnavailable
from bomber.framework.market.basic.base import DataType, MarketDataFeed, make_bar


class FeatureQueryBarFeed(MarketDataFeed):
    def __init__(self, cache, *, key, projection, replay=False, max_market_age_seconds=300):
        super().__init__(cache.source_id)
        if key not in cache.keys or not projection:
            raise ValueError("Bar投影须配置真实OHLCV及已绑定业务键")
        self.cache, self.key, self.projection = cache, key, projection
        self.clock_ns, self.hub = cache.clock_ns, cache.hub
        self.replay = replay
        if not math.isfinite(max_market_age_seconds) or max_market_age_seconds <= 0:
            raise ValueError("市场年龄预算须为有限正数")
        self.max_age_ns = int(max_market_age_seconds * 1e9)
        fields = projection["fields"]
        schema_fields = {f.name: f for f in cache.spec.payload_fields}
        if set(fields) != {"open", "high", "low", "close", "volume"} or any(
                name not in schema_fields or schema_fields[name].kind not in {"decimal", "float", "int"}
                or schema_fields[name].nullable for name in fields.values()):
            raise ValueError("Bar投影须完整绑定不可空的数值OHLCV字段")
        units = {"SECOND": 1_000_000_000, "MINUTE": 60_000_000_000,
            "HOUR": 3_600_000_000_000, "DAY": 86_400_000_000_000}
        try:
            step, unit = projection["bar_spec"].split("-")
            if int(step) <= 0 or int(step)*units[unit] != cache.spec.frequency.period_ns:
                raise ValueError()
        except (ValueError, KeyError, AttributeError):
            raise ValueError("Bar投影频率须与特征Schema一致") from None
        self.requirement = FeatureRequirement("value", cache.dataset_id, cache.schema_version, key, alignment=Alignment.EXACT)
        self._emitted, self._lock = {}, RLock()
        self._lookups = {}

    def _on_subscription_added(self, request):
        if request.data_type is not DataType.BAR or request.bar_spec != self.projection["bar_spec"]:
            raise ValueError("订阅须匹配声明的Bar类型/频率")
        if str(request.instrument_id.symbol) != self.key:
            raise ValueError("Bar订阅与显式业务键不符")

    def _on_subscription_removed(self, request):
        pass

    def connect(self):
        for subscriptions in self._subscriptions.values():
            for request in subscriptions:
                self._on_subscription_added(request)
        self.cache.open()
        self._is_connected = True

    def disconnect(self):
        self._is_connected = False
        self.cache.close()

    @property
    def manifest(self):
        with self._lock:
            return {**self.cache.manifest, "emitted_versions": {str(event): r.revision
                for event, r in self._emitted.items()},
                "minute_lookups": {str(event): dict(value) for event, value in self._lookups.items()}}

    def poll(self, requested_ends, *, force=False):
        # 仅控制线程调用；启动缓存不重放历史Bar，决策等待交由公共帧缓冲。
        with self._lock:
            self.cache.poll(force=force)
            delivered = []
            try:
                for event in sorted(set(requested_ends)):
                    if event in self._emitted:
                        continue
                    if not self.cache.start_ns <= event <= self.cache.end_ns:
                        raise ValueError("Bar请求业务时间不属于配置读取区间")
                    now = self.clock_ns()
                    diagnostic = {"checked_ns": now, "status": "LOOKUP"}
                    self._lookups[event] = diagnostic
                    for old in sorted(self._lookups)[:-32]:
                        del self._lookups[old]
                    try:
                        record = self.hub.feature_snapshot((self.requirement,), FeatureQuery(event, now)).latest("value")
                    except FeatureUnavailable as error:
                        diagnostic.update(status=error.code, detail=str(error),
                            latest_event_ns=self.cache.manifest["latest_event_ns"])
                        if error.code == "MISSING":
                            continue
                        raise
                    if not self.replay and not 0 <= now-record.source_ns <= self.max_age_ns:
                        diagnostic.update(status="SOURCE_STALE", source_ns=record.source_ns)
                        continue
                    diagnostic.update(event_ns=record.ts_event, source_ns=record.source_ns,
                        available_ns=record.available_ns, received_ns=record.received_ns)
                    values = {name: getattr(record.payload, field) for name, field in self.projection["fields"].items()}
                    matched = False
                    for instrument in self._subscriptions:
                        if str(instrument.symbol) != self.key:
                            continue
                        bar = make_bar(instrument_id=instrument, **values, ts_event=record.ts_event,
                            ts_init=record.visible_ns, meta=self.get_instrument_meta(instrument),
                            bar_type=self.projection["bar_spec"])
                        if any(getattr(bar, name).as_decimal() != Decimal(str(value)) for name, value in values.items()):
                            raise ValueError("Bar精度会改变源值；调整显式价格/数量精度")
                        self._emit_bar(bar)
                        matched = True
                    if matched:
                        diagnostic["status"] = "PROJECTED"
                        self._emitted[event] = record
                        delivered.append(event)
                    else:
                        diagnostic["status"] = "NOT_SUBSCRIBED"
                return tuple(delivered)
            except Exception as error:
                self.cache.failure = str(error)
                raise
