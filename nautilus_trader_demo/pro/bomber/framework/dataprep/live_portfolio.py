"""固定多品种主力在线参考适配；读取原子发布缓存。"""

from datetime import datetime
import time
from bomber.framework.market.basic.base import InstrumentId, InstrumentMeta
from .live_cache import LiveReferenceCache


class PortfolioRoleReferences:
    """适配已有多品种LiveFuturesReferences，固定本次路由及累计因子版本。"""
    paths = ()
    refresh_on_event = False

    def __init__(self, service):
        self.service = service
        self.products = service.products
        self.trading_day = service.trading_day
        self.started_ns = service.started_ns
        self.factor_date_basis = service.factor_date_basis
        self._published = LiveReferenceCache()
        self.refresh()

    def validate(self):
        specs = self.specs
        if any(s.venue not in {"SHFE", "INE"} or s.currency != "CNY" for s in specs.values()):
            raise ValueError("组合在线通道仅支持SHFE/INE人民币期货")
        if len(set(self.instrument_ids.values())) != len(self.products):
            raise ValueError("多品种不能映射到同一个真实合约")

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
        return {p: InstrumentId.from_str(f"{s.symbol.lower()}.{s.venue}")
                for p, s in self.specs.items()}

    @property
    def spec(self):
        # 生命周期函数的固定资料比较包含因子，避免窗口混入新版本复权价。
        assignment = self.snapshot(time.time_ns())
        return tuple((p, assignment.instrument(p, "main"), self.specs[p],
                      assignment.factor(p, "main")) for p in self.products)

    @property
    def manifest(self):
        return self._published.manifest

    @property
    def factor_date(self):
        return datetime.fromisoformat(self.manifest["factor_date"]).date()

    def refresh(self):
        def load():
            assignment = self.service.snapshot(time.time_ns())
            specs = {p: self.service.instrument_specs[p]["main"] for p in self.products}
            return assignment, specs, self.service.manifest
        self._published.refresh(load, self.validate)

    def snapshot(self, as_of_ns):
        return self._published.snapshot(as_of_ns)

    def instrument_meta(self, product):
        s = self.specs[product]
        return InstrumentMeta(instrument_id=self.instrument_ids[product],
            price_precision=max(0, -s.tick.normalize().as_tuple().exponent), size_precision=0,
            price_increment=s.tick, multiplier=s.multiplier, currency=s.currency, exchange=s.venue)

