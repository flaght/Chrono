
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from bomber.framework.datahub.sector_roles import SectorDataUnavailable, SectorRoleStore
from bomber.framework.market.basic.base import Bar
from bomber.framework.trader.template import StrategyTemplate

from sector_signal import SectorChainSignal, SectorSignal


@dataclass(frozen=True)
class SectorChainConfig:
    leader_products: tuple[str, str] = ("JM", "I")
    comparison_product: str = "RB"
    signal_role: str = "main"
    execution_product: str = "RB"
    execution_role: str = "secondary"
    target_key: str = "rb_secondary"
    quantity: Decimal = Decimal(1)
    return_period: int = 30
    sector_period: int = 15
    submission_delay_bars: int = 0

    def __post_init__(self) -> None:
        amount = Decimal(str(self.quantity))
        if (not self.target_key or not self.signal_role or not self.execution_role
                or not self.execution_product or not amount.is_finite() or amount <= 0):
            raise ValueError("目标键和正目标手数必须有效")
        if (len(self.leader_products) != 2
                or len(set((*self.leader_products, self.comparison_product))) != 3
                or any(not product for product in (*self.leader_products, self.comparison_product))):
            raise ValueError("信号需两个不同领头品种及一个比较品种")
        if self.execution_product not in self.signal_products:
            raise ValueError("执行品种必须包含在同步信号品种中")
        if min(self.return_period, self.sector_period) < 1 or self.submission_delay_bars < 0:
            raise ValueError("窗口必须为正，执行延迟不能为负")
        object.__setattr__(self, "quantity", amount)

    @property
    def signal_products(self) -> tuple[str, str, str]:
        return (*self.leader_products, self.comparison_product)


@dataclass
class _PendingTarget:
    direction: int
    signal_ns: int
    execution_symbol: str
    remaining_bars: int
    signal: SectorSignal


class SectorChainTargetStrategy(StrategyTemplate):
    """只输出配置的逻辑目标；动态选约、安全换月与真实仓位由框架负责。

    默认同时间戳三个主力真实 Bar 到齐后，以各自主力累计因子生成信号。
    执行品种和角色由配置决定，默认次主力。目标提交前必须收到不早于信号
    的执行合约 Bar；可选再等待 N 根信号之后的执行 Bar。实际成交仍由后端推进。
    """

    def __init__(self, strategy_id: str, data_hub: SectorRoleStore,
                 config: SectorChainConfig | None = None) -> None:
        super().__init__(strategy_id)
        self.data_hub = data_hub
        self.config = config or SectorChainConfig()
        self.signal = SectorChainSignal(
            self.config.return_period, self.config.sector_period,
            leader_products=self.config.leader_products,
            comparison_product=self.config.comparison_product,
        )
        self.complete_frames = 0
        self.unavailable_events = 0
        self.signal_events: list[SectorSignal] = []
        self.execution_events: list[dict[str, object]] = []
        self._frame_ns = -1
        self._closes: dict[str, Decimal] = {}
        self._last_execution_bar_ns: dict[str, int] = {}
        self._pending: _PendingTarget | None = None
        self._committed: tuple[int, str] | None = None

    def on_bar(self, data_key: str, bar: Bar) -> None:
        del data_key
        timestamp = int(bar.ts_event)
        try:
            snapshot = self.data_hub.snapshot(timestamp)
        except SectorDataUnavailable:
            self.unavailable_events += 1
            return
        symbol = str(bar.bar_type.instrument_id.symbol).lower()
        try:
            execution_symbol = snapshot.instrument(
                self.config.execution_product, self.config.execution_role,
            ).lower()
            expected = {
                product: snapshot.instrument(product, self.config.signal_role).lower()
                for product in self.config.signal_products
            }
            factors = {
                product: snapshot.factor(product, self.config.signal_role)
                for product in self.config.signal_products
            }
        except SectorDataUnavailable:
            self.unavailable_events += 1
            return
        if symbol == execution_symbol:
            previous = self._last_execution_bar_ns.get(execution_symbol, -1)
            if timestamp > previous:
                self._last_execution_bar_ns[execution_symbol] = timestamp
                self._advance_pending(timestamp, execution_symbol)

        if timestamp <= self.signal.last_timestamp or timestamp < self._frame_ns:
            return
        if timestamp != self._frame_ns:
            self._frame_ns = timestamp
            self._closes.clear()
        for product in self.config.signal_products:
            if symbol == expected[product]:
                self._closes[product] = Decimal(str(bar.close)) * factors[product]
        if len(self._closes) != len(self.config.signal_products):
            return
        self.complete_frames += 1
        signal = self.signal.update(timestamp, dict(self._closes))
        if signal is None or signal.direction == 0:
            return
        self.signal_events.append(signal)
        desired = (signal.direction, execution_symbol)
        if self._pending is not None and (self._pending.direction, self._pending.execution_symbol) == desired:
            return
        if self._committed == desired:
            # 新信号回到已提交目标时，旧的相反待执行目标必须撤销。
            if self._pending is not None:
                self.execution_events.append({"kind": "target_superseded", "signal_ns": self._pending.signal_ns})
                self._pending = None
            return
        execution_ready = self._last_execution_bar_ns.get(execution_symbol, -1) >= timestamp
        if self.config.submission_delay_bars == 0 and execution_ready:
            self._commit(signal, execution_symbol, timestamp)
        else:
            if self._pending is not None:
                self.execution_events.append({"kind": "target_superseded", "signal_ns": self._pending.signal_ns})
            self._pending = _PendingTarget(
                signal.direction, timestamp, execution_symbol, self.config.submission_delay_bars, signal,
            )
            self.execution_events.append({"kind": "target_queued", "signal_ns": timestamp,
                                          "execution_symbol": execution_symbol, "direction": signal.direction,
                                          "reason": "submission_delay" if self.config.submission_delay_bars
                                          else "waiting_for_execution_bar"})

    def _advance_pending(self, bar_ns: int, execution_symbol: str) -> None:
        pending = self._pending
        if pending is None or bar_ns < pending.signal_ns:
            return
        if execution_symbol != pending.execution_symbol:
            # 执行角色换约后，等待新执行合约自己的 Bar，不借用信号合约时钟。
            pending.execution_symbol = execution_symbol
            pending.remaining_bars = self.config.submission_delay_bars
        # 零延迟允许同时间戳后到的执行行情；正延迟只计信号之后的 Bar。
        if pending.remaining_bars > 0:
            if bar_ns == pending.signal_ns:
                return
            pending.remaining_bars -= 1
        if pending.remaining_bars == 0:
            self._pending = None
            self._commit(pending.signal, execution_symbol, bar_ns)

    def _commit(self, signal: SectorSignal, execution_symbol: str, execution_ns: int) -> None:
        if self._committed == (signal.direction, execution_symbol):
            return
        self.set_target(
            self.config.target_key, self.config.quantity * signal.direction, execution_ns,
            metadata={"signal_ns": signal.ts_event, "target_submission_ns": execution_ns,
                      "fill_semantics": "NEXT_BACKEND_MARKET_EVENT_OR_LATER",
                      "execution_product": self.config.execution_product,
                      "execution_role": self.config.execution_role,
                      "execution_symbol": execution_symbol,
                      "comparison_return": str(signal.comparison_return),
                      "sector_ma": str(signal.sector_ma)},
        )
        self._committed = (signal.direction, execution_symbol)
        self.execution_events.append({"kind": "target_submitted", "signal_ns": signal.ts_event,
                                      "target_submission_ns": execution_ns, "execution_symbol": execution_symbol,
                                      "direction": signal.direction})

    def on_stop(self) -> None:
        if self._pending is not None:
            self.execution_events.append({"kind": "target_expired_no_future_bar",
                                          "signal_ns": self._pending.signal_ns})
            self._pending = None
