"""多品种动态主力执行：同步行情、查询参考快照并提交真实合约目标。"""

from __future__ import annotations

from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

if __package__:
    from .cross_section_signal import CrossSectionMomentumSignal, allocate_group_targets
else:
    from cross_section_signal import CrossSectionMomentumSignal, allocate_group_targets
from bomber.framework.datahub.sector_roles import SectorRoleStore
from bomber.framework.market.basic.base import Bar, InstrumentId
from bomber.framework.trader.template import StrategyTemplate


class MainCrossSectionMomentumStrategy(StrategyTemplate):
    """按品种同步当日主力 Bar，换月时把旧真实合约目标归零。"""

    def __init__(self, strategy_id: str, *, products: tuple[str, ...],
                 roles: SectorRoleStore,
                 instruments: Mapping[str, InstrumentId],
                 multipliers: Mapping[InstrumentId, Decimal],
                 lookback: int, rebalance_interval: int,
                 target_notional: Decimal, group_fraction: Decimal = Decimal("0.30"),
                 target_quantity_cap: Decimal | None = None, require_full_groups: bool = False) -> None:
        super().__init__(strategy_id)
        if len(products) < 2 or len(set(products)) != len(products):
            raise ValueError("至少需要两个不同品种")
        target_notional = Decimal(str(target_notional))
        if lookback < 1 or rebalance_interval < 1 or not target_notional.is_finite() or target_notional <= 0:
            raise ValueError("回看、调仓间隔和目标名义金额必须为正")
        self.products = products
        self.roles = roles
        self.instruments = dict(instruments)
        self.multipliers = dict(multipliers)
        self.lookback = lookback
        self.rebalance_interval = rebalance_interval
        self.target_notional = target_notional
        if target_quantity_cap is not None:
            target_quantity_cap = Decimal(str(target_quantity_cap))
            if (not target_quantity_cap.is_finite() or target_quantity_cap <= 0 or
                    target_quantity_cap != target_quantity_cap.to_integral_value()):
                raise ValueError("目标手数上限须为正整数")
        self.target_quantity_cap = target_quantity_cap
        self.require_full_groups = require_full_groups
        # 目标金额表示每侧总预算，与入选品种数量无关。
        self.signal = CrossSectionMomentumSignal(products, lookback, group_fraction)
        self.latest: dict[str, Bar] = {}
        self.current_main: dict[str, str] = {}
        self.synchronized_frames = 0
        self.rebalances = 0
        self.last_targets: Mapping[str, Decimal] | None = None
        self.last_signal = None
        self.last_emitted_ns = -1
        self.roll_pending = False
        self.order_updates_received = 0
        self.fills_received = 0
        self.last_order_update = None
        self.fill_positions = {}

    def on_order(self, event) -> None:
        self.order_updates_received += 1
        self.last_order_update = event

    def on_fill(self, event) -> None:
        self.fills_received += 1
        # 归属仓位已由公共执行链更新；回调仅观察，不重复入账。
        self.fill_positions = {str(item): self.position(str(item))
                               for item in self.instruments.values()}

    def on_bar(self, data_key: str, bar: Bar) -> None:
        """同步当前主力行情，在满足预热和调仓条件后提交全合约目标。"""
        if data_key != str(bar.bar_type.instrument_id):
            raise ValueError(f"Bar与data_key不匹配: {data_key}")
        snapshot = self.roles.snapshot(bar.ts_event)
        # 按行情事件时刻查询当时已经生效且可见的主力合约与累计因子。
        symbol = str(bar.bar_type.instrument_id.symbol).lower()
        product = next((item for item in self.products
                        if snapshot.instrument(item, "main").lower() == symbol), None)
        if product is None:
            return  # 换月日旧主力 Bar 只用于模拟平仓价格。
        previous = self.current_main.get(product)
        if previous is not None and previous != symbol:
            # 复权窗口保持连续；主力变化后在下一有效同步帧触发调仓。
            self.roll_pending = True
            self.latest.pop(product, None)
        self.current_main[product] = symbol
        old = self.latest.get(product)
        if old is not None and bar.ts_event <= old.ts_event:
            if bar.ts_event < old.ts_event:
                raise ValueError(f"{product} Bar时间回退")
            return
        self.latest[product] = bar
        timestamp = bar.ts_event
        if timestamp <= self.last_emitted_ns or len(self.latest) != len(self.products) or any(
            self.latest[item].ts_event != timestamp for item in self.products
        ):
            return
        self.last_emitted_ns = timestamp
        self.synchronized_frames += 1
        # 只有全部主力同时间戳到齐才推进窗口；复权价仅用于信号。
        signal = self.signal.update({
            item: self.latest[item].close.as_decimal() * snapshot.factor(item, "main")
            for item in self.products
        })
        if signal is None:
            return
        if self.synchronized_frames % self.rebalance_interval and not self.roll_pending:
            # 普通帧遵守调仓间隔，换约帧在窗口已预热后优先处理。
            return
        # 每次提交全合约目标，旧主力与未入选品种均归零。
        targets = {str(instrument): Decimal(0) for instrument in self.instruments.values()}
        current_instruments = {item: self.instruments[snapshot.instrument(item, "main").lower()]
                               for item in self.products}
        raw_prices = {item: self.latest[item].close.as_decimal() for item in self.products}
        multipliers = {item: self.multipliers[current_instruments[item]] for item in self.products}
        product_targets = allocate_group_targets(signal, self.target_notional, raw_prices, multipliers)
        if self.target_quantity_cap is not None:
            product_targets = {p: max(-self.target_quantity_cap, min(self.target_quantity_cap, q))
                               for p, q in product_targets.items()}
        missing = tuple(p for p in (*signal.long_products, *signal.short_products)
                        if product_targets[p] == 0)
        if self.require_full_groups and missing:
            # 新组合不可执行时回到空仓；旧仓退出仍通过完整零目标执行。
            product_targets = {p: Decimal(0) for p in product_targets}
        for item, quantity in product_targets.items():
            targets[str(current_instruments[item])] = quantity
        self.set_targets(targets, timestamp, metadata={
            "long_products": signal.long_products, "short_products": signal.short_products,
            "group_fraction": str(self.signal.group_fraction),
            "group_size": self.signal.group_size,
            "side_target_notional": str(self.target_notional),
            "target_quantity_cap": str(self.target_quantity_cap) if self.target_quantity_cap is not None else None,
            "require_full_groups": self.require_full_groups,
            "incomplete_group_products": missing,
            "product_targets": {item: str(value) for item, value in product_targets.items()},
            "scores": {item: str(value) for item, value in signal.scores.items()},
            "main_contracts": dict(self.current_main),
            "synchronized_frame": self.synchronized_frames,
        })
        self.last_targets = MappingProxyType(targets)
        self.last_signal = signal
        self.rebalances += 1
        self.roll_pending = False
