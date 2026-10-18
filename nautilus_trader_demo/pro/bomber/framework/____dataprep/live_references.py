"""组合在线期货参考资料；来源可替换，不连接行情或交易端。"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
import hashlib
from threading import RLock
import time
from types import MappingProxyType
from collections.abc import Mapping
from copy import deepcopy

from bomber.framework.datahub.sector_roles import SectorDataUnavailable, SectorRoleAssignment
from .factors import parse_cumulative_factors
from .metadata import contract_rows, future_spec_from_basic, parse_future_terms
from .references import parse_role_assignments, ROLE_COLUMNS
from .sources import ReferenceQuery


def live_factor_policy(date_basis, availability=None):
    """区分已对齐适用日与原始来源日；默认发布时间政策随口径明确选择。"""
    if date_basis not in {"trading", "source"}:
        raise ValueError("在线因子日期口径须为 trading 或 source")
    availability = availability or ("aligned" if date_basis == "trading" else "source-day-end")
    allowed = {"aligned", "explicit"} if date_basis == "trading" else {"source-day-end", "explicit", "observed-on-read"}
    if availability not in allowed:
        raise ValueError(f"{date_basis}口径的在线因子可用性须为{sorted(allowed)}")
    return availability


class LiveFuturesReferences:
    """单交易日的多品种/多角色资料服务，显式区分因子来源日与适用日。

    交易日由调用方的TD会话提供。source因子与此前角色来源日一致；
    trading因子已对齐本次交易日。原直接调用默认trading保持兼容。
    observed-on-read显式允许稳定读取的来源日资料从首次观测完成时使用，
    不声称其历史发布时间；真实available_ns仍优先限制可用时刻。
    不推断夜盘日期、不构造缺失因子、不提供历史重放。
    数据源由调用方打开/关闭；所有策略读取只经 snapshot 接口。
    """

    def __init__(self, source, *, products, trading_day, started_ns,
                 roles=("main",), factor_roles=("main",), factor_date_basis="trading", factor_availability=None,
                 refresh_seconds=5, clock_ns=None):
        self.source = source
        self.products = tuple(dict.fromkeys(p.strip().upper() for p in products))
        if not self.products or any(not p.isalpha() for p in self.products):
            raise ValueError("品种不能为空且须为字母")
        self.trading_day = (datetime.strptime(trading_day, "%Y%m%d").date()
                            if isinstance(trading_day, str) else trading_day)
        if type(self.trading_day) is not date or type(started_ns) is not int or started_ns < 0:
            raise ValueError("交易日或启动时间无效")
        if not 0 <= refresh_seconds <= 60:
            raise ValueError("资料刷新间隔须在0至60秒之间")
        self.started_ns = started_ns
        self.factor_date_basis = factor_date_basis
        self.factor_availability = live_factor_policy(factor_date_basis, factor_availability)
        self.roles = self._roles(roles)
        self.factor_roles = self._roles(factor_roles, allow_empty=True)
        if any(not set(self.factor_roles[p]) <= set(self.roles[p]) for p in self.products):
            raise ValueError("因子角色须属于已请求角色")
        self._clock_ns = clock_ns or (lambda: time.time_ns())
        self._refresh_ns = int(refresh_seconds * 1_000_000_000)
        self._last_attempt_ns = None
        self._failure = None
        self._fingerprint = None
        self._assignment = None
        self.instrument_specs = MappingProxyType({})
        self._manifest = {}
        self._lock = RLock()
        self.refresh(force=True)

    def _roles(self, value, *, allow_empty=False):
        result = {p: tuple(value[p] if isinstance(value, Mapping) else value) for p in self.products}
        if any((not items and not allow_empty) or len(set(items)) != len(items)
               or not set(items) <= ROLE_COLUMNS.keys() for items in result.values()):
            raise ValueError("角色须为不重复的 main/secondary/near/far")
        return result

    def _read_bundle(self):
        roles = self.source.contract_structure(ReferenceQuery(products=self.products,
            end_date=self.trading_day - timedelta(days=1)))
        _, source_day = self._select_roles(roles)
        factor_day = source_day if self.factor_date_basis == "source" else self.trading_day
        factors = None
        if any(self.factor_roles.values()):
            factors = self.source.adjustment_factors(ReferenceQuery(products=self.products,
                start_date=factor_day, end_date=factor_day))
        basics = self.source.futures_basic(ReferenceQuery(products=self.products,
            active_on=self.trading_day))
        return roles, factors, basics

    def _select_roles(self, batch):
        roles = parse_role_assignments(batch.frame().rename(columns={"date": "trade_date"}),
            self.products, self.roles, source=batch.source)
        selected = {p: roles.loc[roles.code.eq(p)].iloc[-1] for p in self.products}
        source_days = {row.source_day for row in selected.values()}
        if len(source_days) != 1:
            raise SectorDataUnavailable("多品种角色资料来源日不一致")
        return selected, source_days.pop()

    @staticmethod
    def _bundle_fingerprint(bundle):
        return hashlib.sha256("|".join(b.fingerprint if b else "unused"
                                      for b in bundle).encode()).hexdigest()

    def refresh(self, *, force=False):
        with self._lock:
            now = self._clock_ns()
            if not force and self._last_attempt_ns is not None and 0 <= now - self._last_attempt_ns < self._refresh_ns:
                if self._failure:
                    raise SectorDataUnavailable(self._failure)
                return
            self._last_attempt_ns = now
            # 读取失败后仍保留旧对象用于诊断，但不会将其返回给策略。
            self._failure = "在线参考资料刷新未完成"
            try:
                first = self._read_bundle()
                second = self._read_bundle()
                fingerprint = self._bundle_fingerprint(first)
                if fingerprint != self._bundle_fingerprint(second):
                    raise SectorDataUnavailable("读取期间角色、因子或条款变化，请等待完整更新")
                observed_at = self._clock_ns()
                # 同一内容的首次观测时刻不能随刷新向后移动；新内容不得追溯可见。
                first_observed = (self._manifest["first_observed_ns"]
                    if self._assignment is not None and fingerprint == self._fingerprint else observed_at)
                roles_batch, factors_batch, basics_batch = second
                selected, source_day = self._select_roles(roles_batch)
                factor_day = source_day if self.factor_date_basis == "source" else self.trading_day
                keys = {(factor_day, p, role) for p in self.products for role in self.factor_roles[p]}
                missing_label = "角色来源日" if self.factor_date_basis == "source" else "本次交易日"
                if keys and not factors_batch.rows:
                    raise SectorDataUnavailable(f"缺少{missing_label}累计因子: {sorted(keys)}")
                # 上游提供实际发布时间时优先使用，不让日终假设覆盖未来available_ns。
                parse_availability = ("explicit" if self.factor_availability == "source-day-end"
                    and factors_batch and "available_ns" in factors_batch.columns else self.factor_availability)
                factor_frame = factors_batch.frame() if factors_batch else None
                if self.factor_availability == "observed-on-read":
                    # 仅在线组装填入观测下界，不改写源表或通用历史解析规则。
                    parse_availability = "explicit"
                    if factor_frame is not None and "available_ns" not in factor_frame:
                        factor_frame["available_ns"] = first_observed
                factors = (parse_cumulative_factors(factor_frame, date_basis=self.factor_date_basis,
                    availability=parse_availability, required_keys=keys, source=factors_batch.source)
                    if factors_batch else {})
                if keys - factors.keys():
                    raise SectorDataUnavailable(f"缺少{missing_label}累计因子: {sorted(keys - factors.keys())}")
                # 初次加载沿用旧文件入口的启动时刻；更新版不能对较早事件追溯可见。
                effective = (self.started_ns if self._assignment is None else
                             self._assignment.effective_ns if fingerprint == self._fingerprint else
                             max(self.started_ns, observed_at))
                available = (max(effective, first_observed)
                             if self.factor_availability == "observed-on-read" else effective)
                contracts, cumulative = {}, {}
                for p, row in selected.items():
                    contracts[p] = {r: str(row[ROLE_COLUMNS[r]]).strip().lower() for r in self.roles[p]}
                    cumulative[p] = {}
                    if "available_ns" in row:
                        raw = Decimal(str(row.available_ns))
                        if not raw.is_finite() or raw < 0 or raw != raw.to_integral_value():
                            raise ValueError("角色 available_ns 须为非负整数")
                        available = max(available, int(raw))
                    for role in self.factor_roles[p]:
                        symbol, factor, published = factors[(factor_day, p, role)]
                        contracts[p][role] = symbol  # 外部因子的同角色合约优先。
                        cumulative[p][role] = factor
                        available = max(available, published)
                symbols = {s.upper() for values in contracts.values() for s in values.values()}
                rows = contract_rows(basics_batch.frame(), symbols)
                specs = {}
                for p, values in contracts.items():
                    specs[p] = {}
                    for role, symbol in values.items():
                        # 在线资料自己门控发布时刻；不能调用拒绝版本字段的静态入口。
                        spec = future_spec_from_basic(parse_future_terms(rows[symbol.upper()]))
                        if spec.product != p or not spec.listed <= self.trading_day <= spec.last_trade:
                            raise ValueError("角色合约的品种或生命周期不匹配")
                        specs[p][role] = spec
                        if "available_ns" in rows[symbol.upper()]:
                            raw = Decimal(str(rows[symbol.upper()]["available_ns"]))
                            if not raw.is_finite() or raw < 0 or raw != raw.to_integral_value():
                                raise ValueError("合约条款 available_ns 须为非负整数")
                            available = max(available, int(raw))
                assignment = SectorRoleAssignment(self.trading_day, source_day, effective,
                    available, contracts, cumulative)
                manifest = {"backend": type(self.source).__name__, "trading_day": str(self.trading_day),
                    "factor_date_basis": self.factor_date_basis, "factor_date": str(factor_day),
                    "factor_availability": self.factor_availability,
                    "factor_publication_policy": ("observed-on-read"
                        if self.factor_availability == "observed-on-read" else parse_availability),
                    "role_source_day": str(source_day),
                    "reference_date_selection": "latest_role_source_before_td_requires_upstream_completeness",
                    "role_date_basis": "previous_source_day", "fingerprint": fingerprint,
                    "observed_at_ns": observed_at, "first_observed_ns": first_observed, "effective_ns": effective,
                    "available_ns": available, "evidence": ("stable_read_observation_not_publication_version"
                        if self.factor_availability == "observed-on-read" else "content_hash_not_publication_version"),
                    "sources": {b.dataset.value: {"source": b.source, "sha256": b.fingerprint,
                                                  "rows": len(b.rows)} for b in second if b}}
                self.instrument_specs = MappingProxyType({p: MappingProxyType(values) for p, values in specs.items()})
                self._assignment, self._fingerprint, self._manifest = assignment, fingerprint, manifest
                self._failure = None
            except Exception:
                self._failure = "在线参考资料刷新失败；旧快照不可用于交易"
                raise

    @property
    def manifest(self):
        with self._lock:
            return {**deepcopy(self._manifest), "ready": self._failure is None, "failure": self._failure}

    def snapshot(self, as_of_ns):
        with self._lock:
            self.refresh()
            assignment = self._assignment
            if type(as_of_ns) is not int or as_of_ns < 0:
                raise ValueError("查询时间须为非负整数纳秒")
            if as_of_ns < max(assignment.effective_ns, assignment.available_ns):
                raise SectorDataUnavailable("当前角色或因子版本尚未生效或发布")
            return assignment
