"""单品种单角色在线参考适配；文件和ReferenceSource共享策略读取接口。"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

from bomber.framework.datahub.sector_roles import SectorDataUnavailable, SectorRoleAssignment
from bomber.framework.market.basic.base import InstrumentId, InstrumentMeta
from .factors import load_cumulative_factors
from .metadata import contract_rows, future_spec
from .references import load_role_assignments, ROLE_COLUMNS
from .session import read_feather
from .live_references import LiveFuturesReferences, live_factor_policy


class FileRoleReferences:
    """单交易日文件适配；缺失、冲突及未发布资料不返回旧快照。

    role显式选择main/secondary/near/far。交易日由会话提供，日期和因子
    规则沿用公共解析器；通道允许的交易所及币种由组装层明确配置。
    """

    def __init__(self, *, product, trading_day, contract_struct, factors, fut_basic,
                 started_ns, role="main", factor_date_basis="trading", factor_availability=None,
                 allowed_venues=None, required_currency=None):
        self.product = product.strip().upper()
        if not self.product.isalpha():
            raise ValueError("品种代码须为字母")
        self.role = role
        if role not in ROLE_COLUMNS:
            raise ValueError("角色须为main/secondary/near/far")
        self.allowed_venues = None if allowed_venues is None else frozenset(allowed_venues)
        self.required_currency = required_currency
        self.trading_day = datetime.strptime(trading_day, "%Y%m%d").date()
        self.paths = tuple(Path(p).expanduser().resolve()
                           for p in (contract_struct, factors, fut_basic))
        self.started_ns = started_ns
        self.factor_date_basis = factor_date_basis
        self.factor_availability = live_factor_policy(factor_date_basis, factor_availability)
        if self.factor_availability == "observed-on-read":
            raise ValueError("observed-on-read仅用于在线ReferenceSource，不适用于文件资料")
        self._signature = None
        self._assignment = None
        self.spec = None
        self.refresh()

    def _validate_channel(self, spec):
        if self.allowed_venues is not None and spec.venue not in self.allowed_venues:
            raise ValueError("角色合约交易所不在本次通道允许范围")
        if self.required_currency is not None and spec.currency != self.required_currency:
            raise ValueError("角色合约币种不符合本次通道配置")

    def refresh(self):
        signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in self.paths)
        if signature == self._signature:
            return
        roles = load_role_assignments(self.paths[0], (self.product,), (self.role,))
        past = roles.loc[roles.source_day < self.trading_day]
        if past.empty:
            raise SectorDataUnavailable("本次交易日没有此前来源日的角色资料")
        row = past.iloc[-1]
        factor_day = row.source_day if self.factor_date_basis == "source" else self.trading_day
        key = (factor_day, self.product, self.role)
        availability = self.factor_availability
        if availability == "source-day-end" and "available_ns" in read_feather(self.paths[1]):
            availability = "explicit"
        factors = load_cumulative_factors(self.paths[1], date_basis=self.factor_date_basis,
            availability=availability, required_keys={key})
        if key not in factors:
            label = "角色来源日" if self.factor_date_basis == "source" else "本次交易日"
            raise SectorDataUnavailable(f"缺少{label}累计因子: {key}")
        symbol, factor, factor_available = factors[key]
        basic = read_feather(self.paths[2])
        spec = future_spec(contract_rows(basic, {symbol})[symbol])
        if spec.product != self.product or not spec.listed <= self.trading_day <= spec.last_trade:
            raise ValueError("角色合约品种或上市／到期日期不匹配")
        self._validate_channel(spec)
        available = self.started_ns
        if "available_ns" in row:
            raw = Decimal(str(row["available_ns"]))
            if not raw.is_finite() or raw < 0 or raw != raw.to_integral_value():
                raise ValueError("角色 available_ns 须为非负整数")
            available = max(available, int(raw))
        assignment = SectorRoleAssignment(self.trading_day, row.source_day, self.started_ns,
            max(available, factor_available), {self.product: {self.role: symbol}},
            {self.product: {self.role: factor}})
        if signature != tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in self.paths):
            raise RuntimeError("读取期间参考文件变化，请等待上游原子更新完成后重启")
        self.spec = spec
        self._assignment = assignment
        self.factor_date = factor_day
        self._signature = signature

    @property
    def instrument_id(self):
        return InstrumentId.from_str(f"{self.spec.symbol.lower()}.{self.spec.venue}")

    def instrument_meta(self):
        tick = self.spec.tick
        return InstrumentMeta(instrument_id=self.instrument_id,
            price_precision=max(0, -tick.normalize().as_tuple().exponent), size_precision=0,
            price_increment=tick, multiplier=self.spec.multiplier,
            currency=self.spec.currency, exchange=self.spec.venue)

    def snapshot(self, as_of_ns):
        self.refresh()
        if as_of_ns < self._assignment.effective_ns or as_of_ns < self._assignment.available_ns:
            raise SectorDataUnavailable("本次角色／累计因子尚未生效或发布")
        return self._assignment


class SourceRoleReferences(FileRoleReferences):
    """任意ReferenceSource的单角色薄适配，不依赖具体数据库或策略。"""

    def __init__(self, source, *, product, trading_day, started_ns, role="main",
                 factor_date_basis="source", factor_availability=None, refresh_seconds=5,
                 clock_ns=None, allowed_venues=None, required_currency=None):
        self.product = product.strip().upper()
        self.role = role
        self.allowed_venues = None if allowed_venues is None else frozenset(allowed_venues)
        self.required_currency = required_currency
        self.started_ns = started_ns
        self.paths = ()
        self._references = LiveFuturesReferences(source, products=(self.product,),
            roles=(role,), factor_roles=(role,), trading_day=trading_day, started_ns=started_ns,
            factor_date_basis=factor_date_basis, factor_availability=factor_availability,
            refresh_seconds=refresh_seconds, clock_ns=clock_ns)
        self.trading_day = self._references.trading_day
        self.factor_date_basis = self._references.factor_date_basis
        self.factor_availability = self._references.factor_availability
        self.refresh()

    def refresh(self):
        self._references.refresh()
        spec = self._references.instrument_specs[self.product][self.role]
        self._validate_channel(spec)
        self.spec = spec
        self.factor_date = datetime.fromisoformat(self._references.manifest["factor_date"]).date()

    @property
    def manifest(self):
        return self._references.manifest

    def snapshot(self, as_of_ns):
        self.refresh()
        return self._references.snapshot(as_of_ns)
