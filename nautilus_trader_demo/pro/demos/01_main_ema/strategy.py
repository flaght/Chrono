"""品种主力 EMA 信号；连续复权价仅用于信号，真实合约由动态路由成交。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from bomber.indicators import ExponentialMovingAverage

from bomber.framework.datahub.sector_roles import SectorDataUnavailable, SectorRoleAssignment
from bomber.framework.market.basic.base import Bar
from bomber.framework.trader.execution.events import FillEvent, OrderUpdateEvent
from bomber.framework.trader.template import StrategyTemplate


class RoleSnapshotPort(Protocol):
    """策略只依赖按事件时间查询角色和因子的能力，不负责读取资料文件。"""
    def snapshot(self, as_of_ns: int) -> SectorRoleAssignment: ...


@dataclass(frozen=True)
class MainEmaConfig:
    """EMA 周期按有效主力 Bar 计数；quantity 是多空目标手数的绝对值。"""
    product: str
    venue: str
    fast_period: int = 3
    slow_period: int = 5
    quantity: Decimal = Decimal(1)
    target_key: str = ""

    def __post_init__(self) -> None:
        quantity = Decimal(str(self.quantity))
        if not 0 < self.fast_period < self.slow_period:
            raise ValueError("EMA 周期须满足 0 < fast_period < slow_period")
        if not quantity.is_finite() or quantity <= 0:
            raise ValueError("目标手数须为正且有限")
        product = self.product.strip().upper()
        venue = self.venue.strip().upper()
        if not product.isalpha() or not venue:
            raise ValueError("品种代码须为字母，交易所不能为空")
        # 逻辑目标不绑定某个月份，实际合约交给动态路由决定。
        target_key = self.target_key.strip() or f"{product.lower()}_main"
        object.__setattr__(self, "product", product)
        object.__setattr__(self, "venue", venue)
        object.__setattr__(self, "target_key", target_key)
        object.__setattr__(self, "quantity", quantity)


class MainEmaStrategy(StrategyTemplate):
    """每根主力真实 Bar 用截至该时刻的复权收盘价更新 EMA。"""

    signal_source = "ema"

    def __init__(self, strategy_id: str, data_hub: RoleSnapshotPort,
                 config: MainEmaConfig) -> None:
        super().__init__(strategy_id)
        self.data_hub = data_hub
        self.config = config
        self.fast = ExponentialMovingAverage(self.config.fast_period)
        self.slow = ExponentialMovingAverage(self.config.slow_period)
        self.bars_used = 0
        self.unavailable_events = 0
        self.last_target: Decimal | None = None
        # 仅在有效主力 Bar 更新后推进，旧主力同时间行情不抢占信号时钟。
        self.last_processed_ns = -1
        self.last_main: str | None = None
        self.order_updates_received = 0
        self.last_order_update: OrderUpdateEvent | None = None
        self.last_order_position: Decimal | None = None
        # 成交回调读取已入账的策略归属；供回测和在线运行报告核对。
        self.fills_received = 0
        self.last_fill_position: Decimal | None = None

    def on_order(self, event: OrderUpdateEvent) -> None:
        # 订单数量是账户母订单的事实；只观察状态，不按累计成交量再次记仓。
        self.order_updates_received += 1
        self.last_order_update = event
        self.last_order_position = self.position(self.config.target_key)

    def on_fill(self, event: FillEvent) -> None:
        self.fills_received += 1
        self.last_fill_position = self.position(self.config.target_key)

    def on_bar(self, data_key: str, bar: Bar) -> None:
        del data_key
        timestamp = int(bar.ts_event)
        if timestamp <= self.last_processed_ns:
            return
        try:
            # 查询当时可见的最终映射：角色与因子冲突时，公共层已按因子表覆盖。
            assignment = self.data_hub.snapshot(timestamp)
            main = assignment.instrument(self.config.product, "main").lower()
            cumulative = assignment.factor(self.config.product, "main")
        except SectorDataUnavailable:
            # 不沿用不可用快照，也不将缺失因子当作 1。
            self.unavailable_events += 1
            return
        # 换月日 Feed 同时包含旧、新合约；只有当前主力参与 EMA。
        if (str(bar.bar_type.instrument_id.symbol).lower() != main
                or str(bar.bar_type.instrument_id.venue).upper() != self.config.venue):
            return
        # 累计因子已对齐当前交易日；复权仅用于信号，不修改真实撮合 Bar。
        adjusted_close = bar.close.as_decimal() * cumulative
        value = float(adjusted_close)
        self.fast.update_raw(value)
        self.slow.update_raw(value)
        self.last_processed_ns = timestamp
        self.last_main = main
        self.bars_used += 1
        # 慢线预热完成后才发出目标；换月不重置 EMA，延续复权研究序列。
        if not self.slow.initialized:
            return
        # 快线等于慢线时也取多头，按当前规则生成 +quantity / -quantity。
        target = self.config.quantity if self.fast.value >= self.slow.value else -self.config.quantity
        if target == self.last_target:
            return
        self.last_target = target
        # 提交目标仓位而非直接订单；执行器计算仓位差，路由负责真实合约及换月。
        self.set_target(
            self.config.target_key, target, timestamp,
            metadata={
                "signal": "LONG" if target > 0 else "SHORT",
                "signal_source": self.signal_source,
                "research_main": main,
                "adjusted_close": str(adjusted_close),
                "fast_ema": str(self.fast.value),
                "slow_ema": str(self.slow.value),
                "signal_ts": timestamp,
            },
        )
