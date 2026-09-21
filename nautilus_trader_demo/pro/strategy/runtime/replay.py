"""当前文件回放 Runner 的运行时适配器。"""

from __future__ import annotations

from typing import Any

from strategy.contracts import RuntimeMode
from strategy.runtime.base import HistoricalRunnerPort


class SimpleReplayRuntime:
    """复用现有FileReplayFeed链路的轻量回放运行时。

    本类只验证行情、策略目标和执行路由，不提供撮合、手续费、保证金或PnL。
    """

    mode = RuntimeMode.HISTORICAL

    def __init__(self, runtime_id: str, runner: HistoricalRunnerPort) -> None:
        if not runtime_id.strip():
            raise ValueError("runtime_id 不能为空")
        if getattr(runner, "mode", None) is not RuntimeMode.HISTORICAL:
            raise ValueError("SimpleReplayRuntime只接受HISTORICAL模式Runner")
        self.runtime_id = runtime_id
        self._runner = runner

    def start(self) -> None:
        self._runner.start()

    def run(self) -> Any:
        return self._runner.run_replay()

    def run_replay(self) -> Any:
        """保留当前调用名称；新代码优先使用统一的run()。"""
        return self.run()

    def stop(self) -> None:
        self._runner.stop()
