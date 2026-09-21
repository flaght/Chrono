"""由Bomber/Nautilus BacktestEngine驱动的正式回测运行时。"""

from __future__ import annotations

from typing import Any

from bomber.backtest.config import BacktestEngineConfig
from bomber.backtest.engine import BacktestEngine

from strategy.contracts import RuntimeMode


class NautilusBacktestRuntime:
    """旧Bridge回测兼容入口。

    新代码使用UnifiedHistoricalRuntime组合Runner与NautilusSimExecutionBackend；
    本类只为既有调用保留，不再承载新的执行能力。
    """

    mode = RuntimeMode.HISTORICAL
    engine_name = "bomber.nautilus.BacktestEngine"

    def __init__(
        self,
        runtime_id: str,
        config: BacktestEngineConfig | None = None,
    ) -> None:
        if not runtime_id.strip():
            raise ValueError("runtime_id 不能为空")
        self.runtime_id = runtime_id
        self.engine = BacktestEngine(config=config or BacktestEngineConfig())
        self._started = False
        self._stopped = False

    def add_venue(self, **kwargs: Any) -> None:
        self._ensure_configurable()
        self.engine.add_venue(**kwargs)

    def add_instrument(self, instrument: Any) -> None:
        self._ensure_configurable()
        self.engine.add_instrument(instrument)

    def add_data(self, data: list[Any], **kwargs: Any) -> None:
        self._ensure_configurable()
        self.engine.add_data(data, **kwargs)

    def add_strategy(self, strategy: Any) -> None:
        self._ensure_configurable()
        self.engine.add_strategy(strategy)

    def start(self) -> None:
        if self._stopped:
            raise RuntimeError("已释放的回测Runtime不能重新启动")
        self._started = True

    def run(self) -> Any:
        if not self._started:
            self.start()
        self.engine.run()
        return self.engine.get_result()

    def stop(self) -> None:
        if self._stopped:
            return
        self.engine.dispose()
        self._stopped = True
        self._started = False

    def _ensure_configurable(self) -> None:
        if self._started or self._stopped:
            raise RuntimeError("Runtime启动或释放后不能修改回测配置")
