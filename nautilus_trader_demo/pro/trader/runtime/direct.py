"""当前实时 Runner 的运行时适配器。"""

from __future__ import annotations

from trader.contracts import RuntimeMode
from trader.runtime.base import RunnerLifecyclePort


class DirectLiveRuntime:
    """将现有LIVE Runner暴露为RuntimePort，不改变其内部行为。"""

    mode = RuntimeMode.LIVE

    def __init__(self, runtime_id: str, runner: RunnerLifecyclePort) -> None:
        if not runtime_id.strip():
            raise ValueError("runtime_id 不能为空")
        if getattr(runner, "mode", None) is not RuntimeMode.LIVE:
            raise ValueError("DirectLiveRuntime只接受LIVE模式Runner")
        self.runtime_id = runtime_id
        self._runner = runner

    def start(self) -> None:
        self._runner.start()

    def stop(self) -> None:
        self._runner.stop()
