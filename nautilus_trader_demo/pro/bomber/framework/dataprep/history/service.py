"""历史启动/恢复协调；阶段选择来源，允许缺失只豁免窗口覆盖条件。"""
from dataclasses import dataclass
from .base import HistoryRequest, HistoryError, HistoryUnavailable, select_bars
from .factory import HistoryProviderFactory
from ..sources.policy import DataSourcePurpose


@dataclass(frozen=True)
class HistoryResult:
    bars: tuple
    missing: tuple[int, ...]
    sources: tuple[str, ...]
    allowed_missing: bool
    unavailable_sources: tuple[str, ...] = ()

    def audit(self):
        return {"required": len(self.bars) + len(self.missing), "loaded": len(self.bars),
            "missing": len(self.missing), "first_missing_ns": self.missing[0] if self.missing else None,
            "last_missing_ns": self.missing[-1] if self.missing else None,
            "sources": self.sources, "allowed_missing": self.allowed_missing,
            "unavailable_sources": self.unavailable_sources}


class HistoryService:
    def __init__(self, *, database_backend="dolphindb", database_config=None,
                 file_path=None, missing_policy="fail", factory=HistoryProviderFactory, price_adjuster=None):
        if database_backend == "file" or missing_policy not in {"fail", "allow"}:
            raise HistoryError("须指定数据库Provider及fail/allow缺失策略")
        self.backend, self.config = database_backend, database_config
        self.file_path, self.missing_policy, self.factory = file_path, missing_policy, factory
        self.last_audit = None
        self.audit_history = []
        if price_adjuster is not None and not callable(price_adjuster):
            raise TypeError("price_adjuster须为可调用的历史价格处理器")
        self.price_adjuster = price_adjuster

    def _read(self, backend, config, request, *, phase):
        purpose = DataSourcePurpose.LIVE_MARKET_HISTORY if phase == "preopen" else DataSourcePurpose.LIVE_RECOVERY
        provider = self.factory.create_for(purpose, backend, config)
        try:
            provider.open()
            return select_bars(provider.read(request), request)
        finally:
            provider.close()

    def load(self, instrument, stamps, *, phase, live_bars=()):
        if phase not in {"preopen", "recovery"}:
            raise HistoryError("历史阶段须为preopen/recovery")
        stamps = tuple(sorted(set(stamps)))
        selected, sources, unavailable = {}, [], []
        if stamps:
            request = HistoryRequest(str(instrument), stamps[0] - 59_999_999_999, stamps[-1] + 1,
                max_rows=max(10000, len(stamps) * 2))
            needed = set(stamps)
            selected.update({bar.ts_event: bar for bar in select_bars(live_bars, request) if bar.ts_event in needed})
            if selected:
                sources.append("live-buffer")
            if phase == "preopen" and self.file_path is not None:
                try:
                    bars = self._read("file", self.file_path, request, phase=phase)
                    for bar in bars:
                        if bar.ts_event in needed and bar.ts_event not in selected:
                            selected[bar.ts_event] = bar
                    sources.append("file")
                except FileNotFoundError:
                    unavailable.append("file-not-found")
            if needed - selected.keys():
                # 缺文件或文件不足才回退数据库；盘中恢复直接数据库。
                if self.config is None:
                    if self.missing_policy == "fail":
                        raise HistoryUnavailable("历史数据库Provider未配置，须提供实际表/字段配置")
                    unavailable.append("database-unconfigured")
                else:
                    self._load_database(request, needed, selected, sources, unavailable, phase)
        missing = tuple(stamp for stamp in stamps if stamp not in selected)
        bars = tuple(selected[key] for key in sorted(selected))
        price_audit = None
        if self.price_adjuster is not None and (not missing or self.missing_policy == "allow"):
            adjusted = tuple(self.price_adjuster(bars))
            if [(bar.instrument_id, bar.ts_event) for bar in adjusted] != [(bar.instrument_id, bar.ts_event) for bar in bars]:
                raise HistoryError("历史价格处理不能增删/改写分钟身份或顺序")
            bars = adjusted
            price_audit = getattr(self.price_adjuster, "last_audit", None)
        result = HistoryResult(bars, missing,
            tuple(sources), self.missing_policy == "allow", tuple(unavailable))
        if stamps or self.last_audit is None:
            self.last_audit = {"phase": phase, **result.audit()}
            if self.price_adjuster is not None:
                self.last_audit["price_adjustment"] = price_audit
            self.audit_history.append(self.last_audit)
            self.audit_history = self.audit_history[-50:]
        if missing and self.missing_policy == "fail":
            raise HistoryUnavailable(f"历史窗口缺少{len(missing)}根完整分钟，首根ts_event={missing[0]}")
        return result

    def _load_database(self, request, needed, selected, sources, unavailable, phase):
        try:
            bars = self._read(self.backend, self.config, request, phase=phase)
            sources.append(self.backend)
            for bar in bars:
                if bar.ts_event in needed:
                    previous = selected.get(bar.ts_event)
                    if previous is not None and any(getattr(previous, field) is not None
                            and getattr(bar, field) is not None and getattr(previous, field) != getattr(bar, field)
                            for field in ("close", "adjusted_close", "open", "high", "low", "volume")):
                        raise HistoryError("文件和数据库同一分钟内容冲突，须核验来源版本")
                    if previous is None:
                        selected[bar.ts_event] = bar
        except HistoryUnavailable:
            unavailable.append(self.backend)
