"""可直接复用的注册样例；实际数据集应固定自己的字段与计算版本。"""
from bomber.framework.datahub.features import (
    Alignment, CommoditySpot, DatasetRegistry, DatasetSpec, FrequencySpec,
    IndexPrice, MinuteSignal, PayloadField,
)
from bomber.framework.datahub.features.contracts import MINUTE_NS


def sample_registry():
    day_ns = 1440 * MINUTE_NS
    return DatasetRegistry((
        DatasetSpec("minute.signal", 1, MinuteSignal, (PayloadField("score", "float"),),
            max_source_age_ns=2 * MINUTE_NS),
        DatasetSpec("index.price", 1, IndexPrice, (PayloadField("close", "decimal"),),
            max_source_age_ns=2 * MINUTE_NS),
        # 上海日频业务日期零点为UTC16:00；报价真实观测/发布时间另存。
        DatasetSpec("commodity.spot.daily", 1, CommoditySpot, (
            PayloadField("price", "decimal"), PayloadField("currency", "str"),
            PayloadField("unit", "str"), PayloadField("region", "str"), PayloadField("grade", "str")),
            frequency=FrequencySpec(period_ns=day_ns, anchor_ns=16*60*MINUTE_NS),
            default_alignment=Alignment.ASOF, max_source_age_ns=4*day_ns),
    ))
