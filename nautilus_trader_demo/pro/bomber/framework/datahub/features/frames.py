"""行情照常进入指标；本组件仅装配一次性决策帧，时钟由运行层推进。"""
from dataclasses import dataclass
from types import MappingProxyType

from .contracts import FeatureError, FeatureQuery, FeatureUnavailable, integer


@dataclass(frozen=True)
class StrategyFrame:
    frame_id: str
    event_ns: int
    decision_ns: int
    market: object
    features: object

    def __post_init__(self):
        object.__setattr__(self, "market", MappingProxyType(dict(self.market)))


@dataclass(frozen=True)
class FrameOutcome:
    frame_id: str
    event_ns: int
    status: str
    frame: StrategyFrame | None = None
    reason: str | None = None


class FeatureFrameAssembler:
    """每个assembler对应一个策略需求；调用方串行调用，不在此线程查询数据库。

    deadline时刻数据允许使用；超过deadline后poll直接跳过，不补发旧分钟目标。
    同时刻必须先发布数据再poll；历史使用同一入口和模拟时钟。
    """
    def __init__(self, hub, requirements, *, wait_ns=2_000_000_000, max_pending=32):
        self.hub, self.requirements = hub, tuple(requirements)
        integer(wait_ns, "wait_ns")
        integer(max_pending, "max_pending", 1)
        if not self.requirements or len({r.alias for r in self.requirements}) != len(self.requirements):
            raise FeatureError("决策需求为空或alias重复")
        self.wait_ns, self.max_pending = wait_ns, max_pending
        self._pending = {}
        self._last_event_ns = -1
        self._clock_ns = -1

    def offer(self, frame_id, event_ns, received_ns, market):
        integer(event_ns, "event_ns")
        integer(received_ns, "received_ns")
        if not frame_id or not market or received_ns < event_ns or received_ns < self._clock_ns:
            raise FeatureError("决策帧身份、市场输入或接收时间无效")
        if event_ns <= self._last_event_ns or any(item[0] == event_ns for item in self._pending.values()):
            raise FeatureError("业务帧不能重复或倒退")
        if frame_id in self._pending or len(self._pending) >= self.max_pending:
            raise FeatureError("决策帧ID重复或缓冲已满")
        self._pending[frame_id] = (event_ns, received_ns, received_ns + self.wait_ns, dict(market))

    def poll(self, now_ns):
        integer(now_ns, "now_ns")
        if now_ns < self._clock_ns:
            raise FeatureError("决策时钟回退")
        self._clock_ns = now_ns
        outcomes = []
        for frame_id, (event, received, deadline, market) in sorted(
            tuple(self._pending.items()), key=lambda item: item[1][0]
        ):
            if now_ns < received:
                break
            if now_ns > deadline:
                outcomes.append(FrameOutcome(frame_id, event, "SKIPPED", reason="DECISION_DEADLINE"))
                del self._pending[frame_id]
                self._last_event_ns = event
                continue
            try:
                snapshot = self.hub.feature_snapshot(self.requirements, FeatureQuery(event, min(now_ns, deadline)))
            except FeatureUnavailable as error:
                if now_ns < deadline:
                    break  # 先处理旧帧，避免指标/决策次序倒退。
                outcome = FrameOutcome(frame_id, event, "SKIPPED", reason=f"{error.code}: {error}")
            else:
                frame = StrategyFrame(frame_id, event, now_ns, market, snapshot)
                outcome = FrameOutcome(frame_id, event, "READY", frame=frame)
            outcomes.append(outcome)
            del self._pending[frame_id]
            self._last_event_ns = event
        return tuple(outcomes)

    @property
    def pending_deadlines(self):
        return tuple(sorted(item[2] for item in self._pending.values()))
