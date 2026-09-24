from dataclasses import dataclass

from strategy import (
    BacktestRuntimePort,
    DirectLiveRuntime,
    ReplayRuntimePort,
    RuntimeMode,
    RuntimePort,
    SimpleReplayRuntime,
)


@dataclass
class _RunnerProbe:
    """只记录调用次数的Runner替身，避免测试依赖行情和交易端。"""

    mode: RuntimeMode
    start_calls: int = 0
    stop_calls: int = 0
    replay_calls: int = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def run_replay(self) -> dict[str, int]:
        self.replay_calls += 1
        return {"events": 3}


def test1_direct_live_runtime() -> None:
    """N1：实时Runtime只委托生命周期，不接管交易客户端实现。"""

    runner = _RunnerProbe(RuntimeMode.LIVE)
    runtime = DirectLiveRuntime("direct-live", runner)
    assert isinstance(runtime, RuntimePort)
    runtime.start()
    runtime.stop()
    assert runner.start_calls == 1
    assert runner.stop_calls == 1
    assert runner.replay_calls == 0
    print("N1通过：DirectLiveRuntime正确委托LIVE Runner生命周期")