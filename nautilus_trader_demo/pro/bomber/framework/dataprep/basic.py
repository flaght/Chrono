"""期货、期权版本化条款的文件适配；不猜测资料发布时间。"""

from bomber.framework.datahub.future_basic import FutureBasic, FutureBasicProvider
from bomber.framework.datahub.option_basic import OptionBasic, OptionBasicProvider

from .session import read_feather


def _records(path, value_type, record_factory):
    frame = read_feather(path)
    if not frame.columns.is_unique:
        raise ValueError("基础条款文件包含重复列")
    return tuple(record_factory(value_type.from_mapping(row), row)
                 for row in frame.to_dict(orient="records"))


def load_future_basic_provider(path, *, record_factory):
    """使用公共读取缓存构造期货条款查询服务。"""
    return FutureBasicProvider(_records(path, FutureBasic, record_factory))


def load_option_basic_provider(path, *, record_factory):
    """使用公共读取缓存构造期权条款查询服务。"""
    return OptionBasicProvider(_records(path, OptionBasic, record_factory))
