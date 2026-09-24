"""第五类目标策略：信号合约与执行合约分离，行情/撮合均由Runner装配。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from datahub.sector_roles import SectorDataUnavailable, SectorRoleStore
from market.basic.base import Bar
from trader.template import StrategyTemplate

from .sector_signal import BlackSectorSignal, SectorSignal


@dataclass(frozen=True)
class BlackSectorConfig:
    leader_products: tuple[str, str] = ("JM", "I")
    comparison_product: str = "RB"
    signal_role: str = "secondary"
    execution_product: str = "RB"
    execution_role: str = "main"
    target_key: str = "rb_main"
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


class BlackSectorTargetStrategy(StrategyTemplate):
    """只输出配置的逻辑目标；动态选约、安全换月与真实仓位由框架负责。

    同时间戳三个次主力真实Bar全部送达后才能决策。默认立即提交目标，
    由HistoricalRuntime保证最早下一行情事件撮合；可选再等待RB主力自己的
    N根执行合约Bar后提交，但实际成交仍至少晚一个Backend行情事件。
    """

    def __init__(self, strategy_id: str, data_hub: SectorRoleStore,
                 config: BlackSectorConfig | None = None) -> None:
        super().__init__(strategy_id)
        self.data_hub = data_hub
        self.config = config or BlackSectorConfig()
        self.signal = BlackSectorSignal(
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
        self._last_main_bar_ns: dict[str, int] = {}
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
            main = snapshot.instrument(
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
        if symbol == main:
            previous = self._last_main_bar_ns.get(main, -1)
            if timestamp > previous:
                self._last_main_bar_ns[main] = timestamp
                self._advance_pending(timestamp, main)

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
        desired = (signal.direction, main)
        if self._pending is not None and (self._pending.direction, self._pending.execution_symbol) == desired:
            return
        if self._committed == desired:
            # 新信号回到已提交目标时，旧的相反待执行目标必须撤销。
            if self._pending is not None:
                self.execution_events.append({"kind": "target_superseded", "signal_ns": self._pending.signal_ns})
                self._pending = None
            return
        if self.config.submission_delay_bars == 0:
            self._commit(signal, main, timestamp)
        else:
            if self._pending is not None:
                self.execution_events.append({"kind": "target_superseded", "signal_ns": self._pending.signal_ns})
            self._pending = _PendingTarget(
                signal.direction, timestamp, main, self.config.submission_delay_bars, signal,
            )
            self.execution_events.append({"kind": "target_queued", "signal_ns": timestamp,
                                          "main": main, "direction": signal.direction})

    def _advance_pending(self, bar_ns: int, main: str) -> None:
        pending = self._pending
        if pending is None or bar_ns <= pending.signal_ns:
            return
        if main != pending.execution_symbol:
            # 等待新主力自己的下一根Bar；旧主力行情不充当新主力的执行时钟。
            pending.execution_symbol = main
            pending.remaining_bars = self.config.submission_delay_bars
        pending.remaining_bars -= 1
        if pending.remaining_bars == 0:
            self._pending = None
            self._commit(pending.signal, main, bar_ns)

    def _commit(self, signal: SectorSignal, main: str, execution_ns: int) -> None:
        if self._committed == (signal.direction, main):
            return
        self.set_target(
            self.config.target_key, self.config.quantity * signal.direction, execution_ns,
            metadata={"signal_ns": signal.ts_event, "target_submission_ns": execution_ns,
                      "fill_semantics": "NEXT_BACKEND_MARKET_EVENT_OR_LATER",
                      "execution_product": self.config.execution_product,
                      "execution_role": self.config.execution_role,
                      "execution_symbol": main,
                      "comparison_return": str(signal.comparison_return),
                      "sector_ma": str(signal.sector_ma)},
        )
        self._committed = (signal.direction, main)
        self.execution_events.append({"kind": "target_submitted", "signal_ns": signal.ts_event,
                                      "target_submission_ns": execution_ns, "main": main,
                                      "direction": signal.direction})

    def on_stop(self) -> None:
        if self._pending is not None:
            self.execution_events.append({"kind": "target_expired_no_future_bar",
                                          "signal_ns": self._pending.signal_ns})
            self._pending = None
