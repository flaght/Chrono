"""Nautilus实时节点Driver协议和TradingNode实现。"""

from __future__ import annotations

import threading
import time
from decimal import Decimal
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from market.basic.base import InstrumentId
from trader.execution.contracts import ExecutionReport, OrderIntent
from trader.execution.events import AccountStateEvent, ActiveOrderSnapshot
from trader.execution.simulation.gateway import NautilusOrderGateway


@runtime_checkable
class NautilusLiveDriverPort(Protocol):
    """Live Backend依赖的最小Nautilus驱动边界。"""

    driver_id: str

    def start(self, report_sink: Callable[[ExecutionReport], None]) -> None: ...

    def stop(self) -> None: ...

    def submit_order(self, order: OrderIntent) -> None: ...

    def cancel_strategy(self, strategy_id: str) -> None: ...

    def reconcile(
        self,
    ) -> Mapping[InstrumentId | str, Decimal | int | float | str]: ...


class NautilusTradingNodeDriver:
    """拥有一个已配置但尚未build的TradingNode。

    调用方负责在构造本Driver前向node注册Binance等Data/Exec Client Factory。
    Driver只添加统一订单Gateway、build节点、启动线程并负责停止释放。
    """

    def __init__(
        self,
        driver_id: str,
        node: Any,
        *,
        reconcile_callback: Callable[
            [], Mapping[InstrumentId | str, Decimal | int | float | str]
        ] | None = None,
        ready_callback: Callable[[], bool] | None = None,
        account_state_callback: Callable[[], AccountStateEvent] | None = None,
        active_orders_callback: Callable[[], ActiveOrderSnapshot] | None = None,
        startup_timeout: float = 30.0,
    ) -> None:
        if not driver_id.strip():
            raise ValueError("driver_id不能为空")
        self.driver_id = driver_id
        self.node = node
        self._reconcile_callback = reconcile_callback
        if startup_timeout <= 0:
            raise ValueError("startup_timeout必须大于0")
        self._ready_callback = ready_callback
        self._account_state_callback = account_state_callback
        self._active_orders_callback = active_orders_callback
        self._startup_timeout = startup_timeout
        self._gateway: NautilusOrderGateway | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self, report_sink: Callable[[ExecutionReport], None]) -> None:
        if self._started:
            return
        gateway = NautilusOrderGateway(self.driver_id, report_sink)
        self.node.trader.add_strategy(gateway)
        self.node.build()
        self._gateway = gateway
        self._thread = threading.Thread(
            target=self.node.run,
            name=f"{self.driver_id}-TradingNode",
            daemon=True,
        )
        self._thread.start()
        if self._ready_callback is not None:
            deadline = time.monotonic() + self._startup_timeout
            while not self._ready_callback():
                if self._thread is not None and not self._thread.is_alive():
                    self.node.dispose()
                    self._gateway = None
                    self._thread = None
                    raise RuntimeError("TradingNode在完成启动前已经退出")
                if time.monotonic() >= deadline:
                    try:
                        self.node.stop()
                    finally:
                        if self._thread is not None and self._thread.is_alive():
                            self._thread.join(timeout=5.0)
                        self.node.dispose()
                        self._gateway = None
                        self._thread = None
                    raise TimeoutError(
                        f"TradingNode在{self._startup_timeout:g}秒内未就绪",
                    )
                time.sleep(0.05)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.node.stop()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=10.0)
        finally:
            self.node.dispose()
            self._started = False

    def submit_order(self, order: OrderIntent) -> None:
        if self._gateway is None:
            raise RuntimeError("TradingNode Driver尚未启动")
        self._gateway.enqueue(order)

    def cancel_strategy(self, strategy_id: str) -> None:
        if self._gateway is not None:
            self._gateway.cancel_strategy_orders(strategy_id)

    def reconcile(self) -> Mapping[InstrumentId | str, Decimal | int | float | str]:
        # TradingNode启动时的原生reconciliation由LiveExecEngineConfig控制。
        # 运行期主动对账由调用方注入，避免猜测不同Bomber版本的私有API。
        if self._reconcile_callback is None:
            raise RuntimeError("未配置运行期reconcile_callback")
        positions = self._reconcile_callback()
        if positions is None:
            raise RuntimeError("reconcile_callback必须返回账户仓位映射")
        return positions

    def reconcile_account_state(self) -> AccountStateEvent:
        """资金必须来自柜台权威查询；不从持仓或模拟余额推算。"""
        if self._account_state_callback is None:
            raise RuntimeError("未配置权威账户资金查询回调")
        state = self._account_state_callback()
        if not isinstance(state, AccountStateEvent):
            raise TypeError("账户资金查询必须返回AccountStateEvent")
        return state

    def query_account(self, native_account_id: Any) -> None:
        """在线程安全的节点Loop上发出只读QueryAccount命令。"""
        if not self._started or self._gateway is None:
            raise RuntimeError("TradingNode Driver尚未启动")
        loop = self.node.get_event_loop()
        if loop is None or not loop.is_running():
            raise RuntimeError("TradingNode事件循环未运行")
        loop.call_soon_threadsafe(self._gateway.query_account, native_account_id)

    def reconcile_active_orders(self) -> ActiveOrderSnapshot:
        """只接受调用方提供的柜台全量查询；不把cache.orders_open当权威。"""
        if self._active_orders_callback is None:
            raise RuntimeError("未配置柜台权威活动订单查询回调")
        snapshot = self._active_orders_callback()
        if not isinstance(snapshot, ActiveOrderSnapshot):
            raise TypeError("活动订单查询必须返回ActiveOrderSnapshot")
        return snapshot
