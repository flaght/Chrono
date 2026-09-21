"""统一Runner与模拟Backend的正式历史运行时。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from strategy.contracts import RuntimeMode
from strategy.runner import UnifiedStrategyRunner
from strategy.runtime.market_adapter import NautilusMarketFeedAdapter


@dataclass(frozen=True)
class UnifiedHistoricalResult:
    """同时保留文件回放统计和Nautilus正式撮合结果。"""

    replay_summary: Any
    backend_result: Any
    market_events_processed: int
    execution_reports_received: int


class UnifiedHistoricalRuntime:
    """以统一积木主链运行一次正式回测。

    Market adapter必须先于Runner挂载Feed回调。这样事件N先推进模拟时钟和
    撮合已有订单，随后才交给StrategyTemplate；事件N产生的新订单最早在
    事件N+1成交，避免依赖偶然回调顺序造成同Bar前视。
    """

    mode = RuntimeMode.HISTORICAL
    engine_name = "bomber.nautilus.NautilusSimExecutionBackend"

    def __init__(
        self,
        runtime_id: str,
        runner: UnifiedStrategyRunner,
        market_adapter: NautilusMarketFeedAdapter,
    ) -> None:
        if not runtime_id.strip():
            raise ValueError("runtime_id不能为空")
        if runner.mode is not RuntimeMode.HISTORICAL:
            raise ValueError("UnifiedHistoricalRuntime只接受HISTORICAL Runner")
        self.runtime_id = runtime_id
        self.runner = runner
        self.market_adapter = market_adapter
        if market_adapter.manage_lifecycle:
            raise ValueError(
                "统一Historical Runtime要求Market adapter使用manage_lifecycle=False",
            )
        self._started = False
        self._stopped = False
        self._result: UnifiedHistoricalResult | None = None

    @property
    def backend(self):
        return self.market_adapter.backend

    @property
    def engine(self):
        return getattr(self.backend, "engine", None)

    @property
    def result(self) -> UnifiedHistoricalResult | None:
        return self._result

    def start(self) -> None:
        if self._started:
            return
        if self._stopped:
            raise RuntimeError("已结束的历史Runtime不能重新启动")
        # 注册顺序属于正确性约束：Backend回调必须排在策略回调之前。
        self.market_adapter.start()
        try:
            self.runner.start()
        except Exception:
            self.market_adapter.stop()
            self._stopped = True
            raise
        self._started = True

    def run(self) -> UnifiedHistoricalResult:
        if self._result is not None:
            return self._result
        self.start()
        try:
            replay_summary = self.runner.run_replay()
            finish = getattr(self.backend, "finish", None)
            if callable(finish):
                finish()
        except Exception:
            self.stop()
            raise
        backend_reports = getattr(self.backend, "reports", None)
        report_count = (
            self.market_adapter.reports_received
            if backend_reports is None
            else len(backend_reports)
        )
        self._result = UnifiedHistoricalResult(
            replay_summary=replay_summary,
            backend_result=self.backend.result(),
            market_events_processed=self.market_adapter.events_processed,
            execution_reports_received=report_count,
        )
        return self._result

    def stop(self) -> None:
        if self._stopped:
            return
        try:
            if self._started:
                self.runner.stop()
        finally:
            self.market_adapter.stop()
            self._started = False
            self._stopped = True
