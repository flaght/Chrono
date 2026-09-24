"""五品种截面动量；不依赖Tick/Feather来源或具体执行客户端。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from types import MappingProxyType
from typing import Mapping

from datahub.sector_roles import SectorRoleStore

from market.basic.base import Bar, InstrumentId
from trader.bar_sync import BarSynchronizer, SynchronizedBarFrame
from trader.template import StrategyTemplate


@dataclass(frozen=True)
class CrossSectionConfig:
    instruments: tuple[InstrumentId, ...]
    contract_multipliers: Mapping[InstrumentId, Decimal]
    size_increments: Mapping[InstrumentId, Decimal] | None = None
    lookback: int = 20
    rebalance_interval: int = 5
    target_notional: Decimal = Decimal("100000")

    def __post_init__(self) -> None:
        instruments = tuple(self.instruments)
        if len(instruments) < 2 or len(set(instruments)) != len(instruments):
            raise ValueError("截面策略至少需要两个不重复的标的")
        if self.lookback < 1 or self.rebalance_interval < 1:
            raise ValueError("回看窗口和调仓间隔必须为正整数")
        target_notional = Decimal(str(self.target_notional))
        if target_notional <= 0:
            raise ValueError("目标名义金额必须大于零")
        multipliers = {
            instrument_id: Decimal(str(multiplier))
            for instrument_id, multiplier in self.contract_multipliers.items()
        }
        if set(multipliers) != set(instruments) or any(value <= 0 for value in multipliers.values()):
            raise ValueError("必须为每个标的配置正合约乘数")
        increments = {
            item: Decimal(str((self.size_increments or {}).get(item, 1)))
            for item in instruments
        }
        if any(value <= 0 for value in increments.values()):
            raise ValueError("数量步长必须为正数")
        object.__setattr__(self, "instruments", instruments)
        object.__setattr__(self, "contract_multipliers", MappingProxyType(multipliers))
        object.__setattr__(self, "size_increments", MappingProxyType(increments))
        object.__setattr__(self, "target_notional", target_notional)


class CrossSectionMomentumStrategy(StrategyTemplate):
    """完整同步帧上计算收益率排名，一次提交所有腿的目标快照。"""

    def __init__(self, strategy_id: str, config: CrossSectionConfig) -> None:
        super().__init__(strategy_id)
        self.config = config
        self.synchronizer = BarSynchronizer(config.instruments)
        self.history = {
            item: deque(maxlen=config.lookback + 1) for item in config.instruments
        }
        self.synchronized_frames = 0
        self.rebalances = 0
        self.last_targets: Mapping[str, Decimal] | None = None

    def on_bar(self, data_key: str, bar: Bar) -> None:
        if data_key != str(bar.bar_type.instrument_id):
            raise ValueError(f"Bar与data_key不匹配: {data_key}")
        frame = self.synchronizer.push(bar)
        if frame is None:
            return
        self._on_frame(frame)

    def _on_frame(self, frame: SynchronizedBarFrame) -> None:
        self.synchronized_frames += 1
        for instrument_id, bar in frame.bars.items():
            self.history[instrument_id].append(bar.close.as_decimal())
        if any(len(values) < self.config.lookback + 1 for values in self.history.values()):
            return
        if self.synchronized_frames % self.config.rebalance_interval:
            return

        scores = {
            item: self.history[item][-1] / self.history[item][0] - 1
            for item in self.config.instruments
        }
        # 分数相同时按InstrumentId排序，保证回放结果不依赖文件/回调顺序。
        ranking = sorted(self.config.instruments, key=lambda item: (-scores[item], str(item)))
        long_id, short_id = ranking[0], ranking[-1]
        targets = {str(item): Decimal(0) for item in self.config.instruments}
        for item, direction in ((long_id, 1), (short_id, -1)):
            price = frame.bars[item].close.as_decimal()
            one_contract = price * self.config.contract_multipliers[item]
            step = self.config.size_increments[item]
            units = (self.config.target_notional / one_contract / step).to_integral_value(
                rounding=ROUND_DOWN,
            )
            # CTP默认一手起；BN允许小数数量，但仍严格落在配置的数量网格上。
            quantity = max(step, units * step)
            targets[str(item)] = quantity * direction

        self.rebalances += 1
        self.last_targets = MappingProxyType(targets)
        self.set_targets(
            targets,
            frame.ts_event,
            metadata={
                "long_instrument": str(long_id),
                "short_instrument": str(short_id),
                "synchronized_frame": self.synchronized_frames,
            },
        )


class MainCrossSectionMomentumStrategy(StrategyTemplate):
    """按品种同步当日主力 Bar，换月时把旧真实合约目标归零。"""

    def __init__(self, strategy_id: str, *, products: tuple[str, ...],
                 roles: SectorRoleStore,
                 instruments: Mapping[str, InstrumentId],
                 multipliers: Mapping[InstrumentId, Decimal],
                 lookback: int, rebalance_interval: int,
                 target_notional: Decimal) -> None:
        super().__init__(strategy_id)
        if len(products) < 2 or len(set(products)) != len(products):
            raise ValueError("至少需要两个不同品种")
        if lookback < 1 or rebalance_interval < 1 or target_notional <= 0:
            raise ValueError("回看、调仓间隔和目标名义金额必须为正")
        self.products = products
        self.roles = roles
        self.instruments = dict(instruments)
        self.multipliers = dict(multipliers)
        self.lookback = lookback
        self.rebalance_interval = rebalance_interval
        self.target_notional = target_notional
        self.history = {product: deque(maxlen=lookback + 1) for product in products}
        self.latest: dict[str, Bar] = {}
        self.current_main: dict[str, str] = {}
        self.synchronized_frames = 0
        self.rebalances = 0
        self.last_targets: Mapping[str, Decimal] | None = None
        self.last_emitted_ns = -1
        self.roll_pending = False

    def on_bar(self, data_key: str, bar: Bar) -> None:
        if data_key != str(bar.bar_type.instrument_id):
            raise ValueError(f"Bar与data_key不匹配: {data_key}")
        snapshot = self.roles.snapshot(bar.ts_event)
        symbol = str(bar.bar_type.instrument_id.symbol).lower()
        product = next((item for item in self.products
                        if snapshot.instrument(item, "main").lower() == symbol), None)
        if product is None:
            return  # 换月日旧主力 Bar 只用于模拟平仓价格。
        previous = self.current_main.get(product)
        if previous is not None and previous != symbol:
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
        for item in self.products:
            factor = snapshot.factor(item, "main")
            self.history[item].append(self.latest[item].close.as_decimal() * factor)
        if any(len(self.history[item]) < self.lookback + 1 for item in self.products):
            return
        if self.synchronized_frames % self.rebalance_interval and not self.roll_pending:
            return
        scores = {item: self.history[item][-1] / self.history[item][0] - 1
                  for item in self.products}
        ranking = sorted(self.products, key=lambda item: (-scores[item], item))
        targets = {str(instrument): Decimal(0) for instrument in self.instruments.values()}
        for item, direction in ((ranking[0], 1), (ranking[-1], -1)):
            current_symbol = snapshot.instrument(item, "main").lower()
            instrument = self.instruments[current_symbol]
            price = self.latest[item].close.as_decimal()
            one_contract = price * self.multipliers[instrument]
            units = (self.target_notional / one_contract).to_integral_value(rounding=ROUND_DOWN)
            targets[str(instrument)] = max(Decimal(1), units) * direction
        self.set_targets(targets, timestamp, metadata={
            "long_product": ranking[0], "short_product": ranking[-1],
            "main_contracts": dict(self.current_main),
            "synchronized_frame": self.synchronized_frames,
        })
        self.last_targets = MappingProxyType(targets)
        self.rebalances += 1
        self.roll_pending = False
