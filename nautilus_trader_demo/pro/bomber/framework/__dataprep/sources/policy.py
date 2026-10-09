"""按用途选择来源；数据库导出与离线执行是不同用途。"""
from enum import Enum


class DataSourcePurpose(str, Enum):
    OFFLINE_REFERENCE = "offline_reference"
    OFFLINE_MARKET_HISTORY = "offline_market_history"
    LIVE_REFERENCE = "live_reference"
    LIVE_MARKET_HISTORY = "live_market_history"
    LIVE_RECOVERY = "live_recovery"
    EXPORT = "export"
    FILE_REFERENCE_TEST = "file_reference_test"


def validate_source(backend, purpose):
    purpose = DataSourcePurpose(purpose)
    name = str(backend).strip().lower()
    local = {DataSourcePurpose.OFFLINE_REFERENCE, DataSourcePurpose.OFFLINE_MARKET_HISTORY,
             DataSourcePurpose.LIVE_MARKET_HISTORY, DataSourcePurpose.FILE_REFERENCE_TEST}
    required = "file" if purpose in local else "dolphindb"
    if name != required:
        raise ValueError(f"{purpose.value}须使用{required}数据源，禁止静默回退")
    return purpose
