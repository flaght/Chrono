"""Live会话契约；行情来源和交易来源分别标识、分别检查。"""

from contextlib import ExitStack
from dataclasses import dataclass, field
import time
from typing import Mapping, Protocol


@dataclass(frozen=True)
class ChannelIdentity:
    domain: str
    channel: str
    source_id: str

    def __post_init__(self):
        if self.domain not in {"market", "execution", "runtime"}:
            raise ValueError("通道用途须为market、execution或runtime")
        if not self.channel.strip() or not self.source_id.strip():
            raise ValueError("渠道与来源标识不能为空")


@dataclass(frozen=True)
class ReadinessIssue:
    source: ChannelIdentity
    code: str
    detail: str
    context: Mapping = field(default_factory=dict)

    def as_dict(self):
        return {"domain": self.source.domain, "channel": self.source.channel,
                "source_id": self.source.source_id, "code": self.code,
                "detail": self.detail, "context": dict(self.context)}


@dataclass(frozen=True)
class ReadinessResult:
    sources: tuple[ChannelIdentity, ...]
    issues: tuple[ReadinessIssue, ...] = ()

    @property
    def ready(self):
        return bool(self.sources) and not self.issues

    def __bool__(self):
        return self.ready

    def as_dict(self):
        return {"ready": self.ready,
                "sources": [dict(vars(source)) for source in self.sources],
                "issues": [issue.as_dict() for issue in self.issues]}


class ChannelReadinessPort(Protocol):
    identity: ChannelIdentity
    def check(self) -> ReadinessResult: ...


class SessionReadiness:
    """组合独立通道，完整收集失败原因；检查失败或异常均保持闭闸。"""
    def __init__(self, *channels: ChannelReadinessPort, profile=None, clock_ns=None):
        if not channels or len({channel.identity for channel in channels}) != len(channels):
            raise ValueError("会话须有明确且不重复的通道来源")
        self.last_result = None
        self.channels = channels
        self.profile = profile
        self.clock_ns = clock_ns or (lambda: time.time_ns())

    def check(self):
        sources, issues = [], []
        for channel in self.channels:
            sources.append(channel.identity)
            try:
                result = channel.check()
                if (result.sources != (channel.identity,) or
                        any(issue.source != channel.identity for issue in result.issues)):
                    raise ValueError("适配器返回了其他来源的就绪结果")
                issues.extend(result.issues)
            except Exception as error:
                issues.append(ReadinessIssue(channel.identity, "CHECK_ERROR", str(error)))
        if self.profile is not None:
            source = ChannelIdentity("runtime", "session", "trading-window")
            try:
                if not self.profile.is_open(self.clock_ns()):
                    sources.append(source)
                    issues.append(ReadinessIssue(source, "OUTSIDE_SESSION", "当前不在配置的运行时段"))
            except Exception as error:
                sources.append(source)
                issues.append(ReadinessIssue(source, "CHECK_ERROR", str(error)))
        self.last_result = ReadinessResult(tuple(sources), tuple(issues))
        return self.last_result

    def __call__(self):
        return bool(self.check())


@dataclass
class LiveRunContext:
    trading_day: str | None
    resources: ExitStack
    references: object = None


class LiveSessionControllerPort(Protocol):
    @property
    def event_label(self) -> str: ...
    def prepare(self, resources: ExitStack) -> LiveRunContext: ...
    def finish_preparation(self, context: LiveRunContext) -> None: ...
    def start(self, session, context: LiveRunContext) -> None: ...
    def poll(self, session) -> bool: ...
    def verify(self, session) -> None: ...
    def shutdown(self, session, context: LiveRunContext | None) -> dict: ...
    def snapshot(self, session, context: LiveRunContext | None) -> dict: ...
    def events(self, session): ...
