"""在线运行积木；渠道适配器显式从channels导入，不隐式加载柜台SDK。"""

from .contracts import (ChannelIdentity, ChannelReadinessPort, LiveRunContext,
                        LiveSessionControllerPort, ReadinessIssue, ReadinessResult,
                        SessionReadiness)
from .profiles import LiveProfile, MarketTimeProfile

__all__ = ["ChannelIdentity", "ChannelReadinessPort", "LiveRunContext",
           "LiveSessionControllerPort", "ReadinessIssue", "ReadinessResult",
           "SessionReadiness", "LiveProfile", "MarketTimeProfile", "ManagedLiveRuntime", "DirectLiveRuntime"]


def __getattr__(name):
    if name in {"ManagedLiveRuntime", "DirectLiveRuntime"}:
        from . import runtime
        return getattr(runtime, name)
    raise AttributeError(name)
