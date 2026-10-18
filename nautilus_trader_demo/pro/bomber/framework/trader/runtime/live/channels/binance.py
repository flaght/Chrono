"""Binance在线适配；行情流与账户/执行连接独立检查，无CTP规则。"""

from ..contracts import ChannelIdentity, ReadinessIssue, ReadinessResult
from ..health import StreamHealthCheck
from ..profiles import MarketTimeProfile


class BinanceMarketChannel(StreamHealthCheck):
    def __init__(self, upstream, *, source_id=None, profile=None, **kwargs):
        profile = profile or MarketTimeProfile()
        if profile.allow_historical_events:
            raise ValueError("Binance在线渠道只支持realtime；历史文件使用独立文件Feed")
        super().__init__(upstream,
            ChannelIdentity("market", "binance", source_id or upstream.source_id),
            profile=profile, **kwargs)


class BinanceExecutionChannel:
    """就绪回调读取已对账状态；权威HTTP查询由控制线程/执行积木负责。"""
    def __init__(self, driver, *, account_ready, source_id=None, connection_ready=None):
        self.driver, self.account_ready = driver, account_ready
        self.connection_ready = connection_ready or driver.node.is_running
        self.identity = ChannelIdentity("execution", "binance", source_id or driver.driver_id)

    def check(self):
        issues = []
        if not self.connection_ready():
            issues.append(ReadinessIssue(self.identity, "EXECUTION_DISCONNECTED", "Binance执行节点未运行"))
        elif not self.account_ready():
            issues.append(ReadinessIssue(self.identity, "ACCOUNT_NOT_READY", "Binance账户尚未完成对账或已失效"))
        return ReadinessResult((self.identity,), tuple(issues))
