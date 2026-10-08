"""在线主力参考资料：复用回测文件，按本次 TD 交易日绑定。"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

from bomber.framework.datahub.sector_roles import SectorDataUnavailable, SectorRoleAssignment
from bomber.framework.dataprep.factors import load_cumulative_factors
from bomber.framework.dataprep.metadata import contract_rows, future_spec
from bomber.framework.dataprep.references import load_role_assignments
from bomber.framework.dataprep.session import read_feather
from bomber.framework.market.basic.base import InstrumentId, InstrumentMeta


class LiveMainReferences:
    """每次查询检查文件版本；资料删除、冲突或尚未发布时不沿用旧值。

    首版只支持一个已核对的交易日，不根据自然日期推断夜盘交易日。
    生效时间采用本次加载起点，启动前未完整观察的分钟不用于交易。
    """

    def __init__(self, *, product, trading_day, contract_struct, factors, fut_basic,
                 started_ns, factor_availability="aligned"):
        self.product = product.strip().upper()
        if not self.product.isalpha():
            raise ValueError("品种代码须为字母")
        self.trading_day = datetime.strptime(trading_day, "%Y%m%d").date()
        self.paths = tuple(Path(p).expanduser().resolve()
                           for p in (contract_struct, factors, fut_basic))
        self.started_ns = started_ns
        self.factor_availability = factor_availability
        self._signature = None
        self._assignment = None
        self.spec = None
        self.refresh()

    def refresh(self):
        signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in self.paths)
        if signature == self._signature:
            return
        roles = load_role_assignments(self.paths[0], (self.product,), ("main",))
        past = roles.loc[roles.source_day < self.trading_day]
        if past.empty:
            raise SectorDataUnavailable("本次交易日没有此前来源日的角色资料")
        row = past.iloc[-1]
        key = (self.trading_day, self.product, "main")
        factors = load_cumulative_factors(
            self.paths[1], date_basis="trading", availability=self.factor_availability,
            required_keys={key})
        if key not in factors:
            raise SectorDataUnavailable(f"缺少本次交易日累计因子: {key}")
        symbol, factor, factor_available = factors[key]
        # 与回测保持相同契约：当前交易日因子表的真实合约优先。
        basic = read_feather(self.paths[2])
        spec = future_spec(contract_rows(basic, {symbol})[symbol])
        if spec.product != self.product or not spec.listed <= self.trading_day <= spec.last_trade:
            raise ValueError("主力合约品种或上市／到期日期不匹配")
        if spec.venue not in {"SHFE", "INE"} or spec.currency != "CNY":
            raise ValueError("首版主力EMA SimNow只支持CNY的SHFE／INE期货")
        available = self.started_ns
        if "available_ns" in row:
            raw = Decimal(str(row["available_ns"]))
            if not raw.is_finite() or raw < 0 or raw != raw.to_integral_value():
                raise ValueError("角色 available_ns 须为非负整数")
            available = max(available, int(raw))
        assignment = SectorRoleAssignment(
            self.trading_day, row.source_day, self.started_ns,
            max(available, factor_available),
            {self.product: {"main": symbol}}, {self.product: {"main": factor}})
        # 只有本次完整读取成功才替换缓存；失败会继续抛错而非返回旧快照。
        if signature != tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in self.paths):
            raise RuntimeError("读取期间参考文件变化，请等待上游原子更新完成后重启")
        self.spec = spec
        self._assignment = assignment
        self._signature = signature

    @property
    def instrument_id(self):
        return InstrumentId.from_str(f"{self.spec.symbol.lower()}.{self.spec.venue}")

    def instrument_meta(self):
        tick = self.spec.tick
        return InstrumentMeta(
            instrument_id=self.instrument_id,
            price_precision=max(0, -tick.normalize().as_tuple().exponent),
            size_precision=0, price_increment=tick, multiplier=self.spec.multiplier,
            currency=self.spec.currency, exchange=self.spec.venue)

    def snapshot(self, as_of_ns):
        self.refresh()
        if as_of_ns < self._assignment.effective_ns or as_of_ns < self._assignment.available_ns:
            raise SectorDataUnavailable("本次角色／累计因子尚未生效或发布")
        return self._assignment
