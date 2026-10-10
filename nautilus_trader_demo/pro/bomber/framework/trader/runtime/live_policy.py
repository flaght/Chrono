"""兼容旧导入；通用时间策略与CTP渠道规则已分离。"""

from .live.channels.ctp import (CtpLiveProfile as LiveTimePolicy, InstrumentHealthGate,
    add_environment_arguments, validate_environment_arguments, validate_replay_environment)
from .live.health import validate_instrument_max_age

__all__ = ["LiveTimePolicy", "InstrumentHealthGate", "add_environment_arguments",
           "validate_environment_arguments", "validate_replay_environment", "validate_instrument_max_age"]
