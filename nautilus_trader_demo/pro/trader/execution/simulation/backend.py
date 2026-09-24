"""以Nautilus BacktestEngine为内核的通用模拟执行Backend。"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from bomber.backtest.config import BacktestEngineConfig
from bomber.backtest.engine import BacktestEngine

from trader.execution.contracts import ExecutionBackendKind, ExecutionReport, OrderIntent
from trader.execution.ports import VenueSimulationProfilePort
from trader.execution.simulation.gateway import NautilusOrderGateway


class NautilusSimExecutionBackend:
    """统一拥有BacktestEngine及其配置、行情推进和释放过程。

    E5b建立引擎所有权和生命周期；E5c通过内部
    ``NautilusOrderGateway``接入订单意图、原生订单和统一执行回报。
    """

    kind = ExecutionBackendKind.SIMULATION

    def __init__(
        self,
        backend_id: str,
        config: BacktestEngineConfig | None = None,
        *,
        engine: Any | None = None,
    ) -> None:
        if not backend_id.strip():
            raise ValueError("backend_id不能为空")
        self.backend_id = backend_id
        self.engine = (
            engine
            if engine is not None
            else BacktestEngine(config=config or BacktestEngineConfig())
        )
        self._profiles: dict[str, VenueSimulationProfilePort] = {}
        self._report_handlers: list[Callable[[ExecutionReport], None]] = []
        self._reports: list[ExecutionReport] = []
        self._gateway: NautilusOrderGateway | None = None
        self._started = False
        self._streaming = False
        self._finalized = False
        self._disposed = False
        self._result: Any = None

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_disposed(self) -> bool:
        return self._disposed

    @property
    def profiles(self) -> tuple[VenueSimulationProfilePort, ...]:
        return tuple(self._profiles.values())

    @property
    def reports(self) -> tuple[ExecutionReport, ...]:
        return tuple(self._reports)

    def add_profile(self, profile: VenueSimulationProfilePort) -> None:
        """注册交易场所Profile，并立即创建对应模拟Venue。"""

        self._ensure_configurable()
        venue_key = str(profile.venue)
        if venue_key in self._profiles:
            raise ValueError(f"模拟Venue已注册: {venue_key}")
        self.engine.add_venue(**dict(profile.build_backend_config()))
        self._profiles[venue_key] = profile

    def add_instrument(self, instrument: Any) -> None:
        self._ensure_configurable()
        self.engine.add_instrument(instrument)

    def add_strategy(self, strategy: Any) -> None:
        self._ensure_configurable()
        self.engine.add_strategy(strategy)

    def add_data(self, data: Sequence[Any], **kwargs: Any) -> None:
        """注册历史行情；行情必须来自market/replay的标准事件。"""

        self._ensure_configurable()
        batch = list(data)
        if not batch:
            raise ValueError("data不能为空")
        self.engine.add_data(batch, **kwargs)

    def start(self) -> None:
        """封闭配置阶段；BacktestEngine会在首次run时启动内部组件。"""

        if self._disposed:
            raise RuntimeError("已释放的Backend不能重新启动")
        if self._finalized:
            raise RuntimeError("已结束的Backend不能重新启动")
        self._ensure_gateway()
        self._started = True

    def run(self) -> Any:
        """一次性运行已注册的历史行情并返回Nautilus回测结果。"""

        self._ensure_runnable()
        self.start()
        self.engine.run()
        self._finalized = True
        self._result = self.engine.get_result()
        return self._result

    def process_market_event(self, event: Any) -> Sequence[ExecutionReport]:
        """以streaming模式推进一个标准行情事件。

        该入口用于后续在线仿真；每次推进后清除已消费批次，唯一模拟时钟仍由
        BacktestEngine掌握。返回值只包含本次事件推进新产生的执行回报。
        """

        self._ensure_runnable()
        report_offset = len(self._reports)
        self.start()
        self.engine.add_data([event])
        self.engine.run(streaming=True)
        self.engine.clear_data()
        self._streaming = True
        return tuple(self._reports[report_offset:])

    def submit_order(self, order: OrderIntent) -> None:
        self._ensure_runnable()
        if order.backend_id != self.backend_id:
            raise ValueError(
                f"订单Backend不匹配: {order.backend_id} != {self.backend_id}",
            )
        gateway = self._ensure_gateway()
        gateway.enqueue(order)

    def register_report_handler(
        self,
        handler: Callable[[ExecutionReport], None],
    ) -> None:
        if handler not in self._report_handlers:
            self._report_handlers.append(handler)

    def cancel_strategy(self, strategy_id: str) -> None:
        if not strategy_id.strip():
            raise ValueError("strategy_id不能为空")
        if self._gateway is not None:
            self._gateway.cancel_strategy_orders(strategy_id)

    def result(self) -> Any:
        return self._result

    def finish(self) -> Any:
        """结束流式撮合并生成结果，但保留Engine供报告查询。

        正式Historical Runtime需要先返回结果和原生报告，随后再由stop统一释放
        Engine。一次性``run()``已经自行结束，不会重复调用``engine.end()``。
        """

        if self._disposed:
            raise RuntimeError("已释放的Backend不能结束运行")
        if self._streaming and not self._finalized:
            self.engine.end()
            self._finalized = True
            self._result = self.engine.get_result()
            self._started = False
        return self._result

    def stop(self) -> None:
        """结束流式运行并释放引擎；重复调用安全。"""

        if self._disposed:
            return
        self.finish()
        self.engine.dispose()
        self._disposed = True
        self._started = False

    def _ensure_configurable(self) -> None:
        if self._started or self._finalized or self._disposed:
            raise RuntimeError("Backend启动、结束或释放后不能修改静态回测配置")

    def _ensure_runnable(self) -> None:
        if self._disposed:
            raise RuntimeError("已释放的Backend不能运行")
        if self._finalized:
            raise RuntimeError("已结束的Backend不能再次运行")

    def _ensure_gateway(self) -> NautilusOrderGateway:
        if self._gateway is None:
            self._gateway = NautilusOrderGateway(self.backend_id, self._receive_report)
            self.engine.add_strategy(self._gateway)
        return self._gateway

    def _receive_report(self, report: ExecutionReport) -> None:
        self._reports.append(report)
        for handler in tuple(self._report_handlers):
            handler(report)
