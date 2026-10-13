"""固定多品种主力在线参考适配；读取原子发布缓存。"""

from dataclasses import replace
from datetime import datetime
import re
import time
from bomber.framework.market.basic.base import InstrumentId, InstrumentMeta
from .live_cache import LiveReferenceCache


CTP_PORTFOLIO_VENUES = frozenset({"SHFE", "INE", "DCE", "CZCE"})


def ctp_futures_symbol(symbol, venue):
    """将已由条款/生命周期核验的真实期货代码转为CTP柜台代码。

    资料身份与原始指纹仍保留在service中；不猜测主力合约或交割年。
    郑商所使用大写品种和YMM，其余三个交易所使用小写品种和YYMM。
    """
    match = re.fullmatch(r"([A-Za-z]+)(\d{3,4})", str(symbol).strip())
    if venue not in CTP_PORTFOLIO_VENUES or not match:
        raise ValueError(f"不支持的CTP真实期货代码: {symbol}.{venue}")
    product, month = match.groups()
    if not 1 <= int(month[-2:]) <= 12 or (venue != "CZCE" and len(month) != 4):
        raise ValueError(f"无效的CTP期货交割月份: {symbol}.{venue}")
    if venue == "CZCE":
        return product.upper() + month[-3:]
    return product.lower() + month


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
        unsupported = {p: (s.venue, s.currency) for p, s in specs.items()
                       if s.venue not in CTP_PORTFOLIO_VENUES or s.currency != "CNY"}
        if unsupported:
            raise ValueError(f"组合在线通道仅支持SHFE/INE/DCE/CZCE人民币期货: {unsupported}")
        if len(set(self.instrument_ids.values())) != len(self.products):
            raise ValueError("多品种不能映射到同一个真实合约")

    def assert_expected(self, expected, source_symbols=None):
        actual = {p: str(i) for p, i in self.instrument_ids.items()}
        if expected and actual != expected:
            raise ValueError(f"本次主力不符: expected={expected} actual={actual}")
        for product, symbol in (source_symbols or {}).items():
            # 四位YYMM手工预期必须核对原始资料，不能截掉十年位后误认为同一合约。
            if self.specs[product].symbol.upper() != symbol.upper():
                raise ValueError(f"本次主力资料代码不符: {product} expected={symbol} "
                                 f"actual={self.specs[product].symbol}")

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
        return {p: InstrumentId.from_str(f"{ctp_futures_symbol(s.symbol, s.venue)}.{s.venue}")
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
            # 只在在线CTP边界适配角色代码，原始资料/因子/条款及指纹不修改。
            # unsupported由validate统一诊断，不提前转换为其他市场的代码。
            contracts = {p: {role: (ctp_futures_symbol(symbol, specs[p].venue).lower()
                                    if specs[p].venue in CTP_PORTFOLIO_VENUES else symbol)
                             for role, symbol in assignment.contracts[p].items()}
                         for p in self.products}
            assignment = replace(assignment, contracts=contracts)
            return assignment, specs, self.service.manifest
        self._published.refresh(load, self.validate)

    def snapshot(self, as_of_ns):
        return self._published.snapshot(as_of_ns)

    def instrument_meta(self, product):
        s = self.specs[product]
        return InstrumentMeta(instrument_id=self.instrument_ids[product],
            price_precision=max(0, -s.tick.normalize().as_tuple().exponent), size_precision=0,
            price_increment=s.tick, multiplier=s.multiplier, currency=s.currency, exchange=s.venue)

