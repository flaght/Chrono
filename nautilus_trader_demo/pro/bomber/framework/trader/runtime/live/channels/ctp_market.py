"""CTP MD独立配置；不创建TraderApi，不要求交易前置、资金或账户锁。"""

from dataclasses import dataclass
from datetime import datetime
import os
from ..local import LocalExecutionLifecycle
from .ctp import CtpLiveProfile, required


@dataclass(frozen=True)
class CtpMarketCredentials:
    broker_id: str
    user_id: str
    password: str
    production_mode: bool


class CtpMarketSessionLifecycle(LocalExecutionLifecycle):
    def __init__(self, *, expected_trading_day, environment="realtime", replay_md_trading_day=None,
                 resume=False, state_file=None):
        if datetime.strptime(expected_trading_day, "%Y%m%d").strftime("%Y%m%d") != expected_trading_day:
            raise ValueError("独立CTP行情须声明有效参考交易日YYYYMMDD")
        super().__init__(expected_trading_day, resume=resume, state_file=state_file)
        self.profile = CtpLiveProfile(environment, replay_md_trading_day)
        self.md_front = required("CTP_MD_ADDRESS")
        self.md_credentials = CtpMarketCredentials(os.getenv("CTP_BROKER_ID", "9999"),
            required("CTP_ACCOUNT_ID"), required("CTP_PASSWORD"),
            os.getenv("CTP_PRODUCTION_MODE", "true").lower() in {"1", "true", "yes", "on"})
