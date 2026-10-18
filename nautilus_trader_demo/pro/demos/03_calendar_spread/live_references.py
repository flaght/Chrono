"""跨期双腿投影；资料读取、可见性和控制线程缓存复用公共层。"""

import time

from bomber.framework.dataprep.live_cache import LiveReferenceCache
from bomber.framework.dataprep.live_portfolio import CTP_PORTFOLIO_VENUES, ctp_futures_symbol
from bomber.framework.market.basic.base import InstrumentId, InstrumentMeta


class CalendarReferences:
    paths = ()
    refresh_on_event = False
    factor_date = None
    # 兼容原生CTP生命周期的报告契约；跨期没有因子日期口径。
    factor_date_basis = None

    def __init__(self, service, product, roles, *, clock_ns=None):
        self.service, self.product, self.roles = service, product, tuple(roles)
        self.trading_day = service.trading_day
        self.clock_ns = clock_ns or time.time_ns
        if service.products != (product,) or tuple(service.roles[product]) != self.roles:
            raise ValueError("跨期资料须为单品种配置的两个角色")
        if len(self.roles) != 2 or any(service.factor_roles.values()):
            raise ValueError("跨期只读取两个角色及真实条款，不读取复权因子")
        self._published = LiveReferenceCache(clock_ns=self.clock_ns)
        self.refresh()

    @property
    def publication_lock(self):
        return self._published.publication_lock

    @publication_lock.setter
    def publication_lock(self, value):
        self._published.publication_lock = value

    @property
    def specs(self):
        return dict(self._published.value[1])

    @property
    def instrument_ids(self):
        return {role: InstrumentId.from_str(f"{ctp_futures_symbol(s.symbol, s.venue)}.{s.venue}")
                for role, s in self.specs.items()}

    @property
    def spec(self):
        assignment = self.snapshot(self.clock_ns())
        return (assignment.source_day, tuple((r, assignment.instrument(self.product, r), self.specs[r])
                for r in self.roles))

    @property
    def manifest(self):
        return {**self._published.manifest, "signal_price_basis": "raw_contract_close",
                "requested_roles": list(self.roles), "adjustment_factors_used": False}

    def snapshot(self, as_of_ns):
        return self._published.snapshot(as_of_ns)

    def refresh(self):
        def load():
            assignment = self.service.snapshot(self.clock_ns())
            return assignment, dict(self.service.instrument_specs[self.product]), self.service.manifest
        self._published.refresh(load, self.validate)

    def validate(self):
        specs = self.specs
        if set(specs) != set(self.roles) or any(s.product != self.product or
                s.venue not in CTP_PORTFOLIO_VENUES or s.currency != "CNY" for s in specs.values()):
            raise ValueError("跨期支持SHFE/INE/DCE/CZCE单品种人民币期货")
        if len({s.venue for s in specs.values()}) != 1 or len(set(self.instrument_ids.values())) != 2:
            raise ValueError("跨期双腿须为同交易所的不同真实合约")

    def assert_expected(self, expected):
        if not expected:
            return
        if set(expected) != set(self.roles):
            raise ValueError("手工预期须完整覆盖两个角色")
        for role, value in expected.items():
            spec = self.specs[role]
            parsed = InstrumentId.from_str(value)
            # CZCE四位资料代码必须核对年份，不能截断十年位后误判相同。
            symbol = str(parsed.symbol)
            canonical = f"{ctp_futures_symbol(symbol, str(parsed.venue))}.{parsed.venue}"
            if canonical != str(self.instrument_ids[role]) or (
                    spec.venue == "CZCE" and len(symbol) - len(self.product) == 4
                    and symbol.upper() != spec.symbol.upper()):
                raise ValueError(f"跨期预期合约不符: {role} expected={value} actual={spec.symbol}.{spec.venue}")

    def instrument_meta(self, role):
        s = self.specs[role]
        return InstrumentMeta(instrument_id=self.instrument_ids[role],
            price_precision=max(0, -s.tick.normalize().as_tuple().exponent), size_precision=0,
            price_increment=s.tick, multiplier=s.multiplier, currency=s.currency, exchange=s.venue)
