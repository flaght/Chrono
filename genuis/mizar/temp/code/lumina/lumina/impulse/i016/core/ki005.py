# -*- encoding:utf-8 -*-
"""
dw005: PLUS 影线差因子 (Premium between Lower and Upper Shadow)

来源: 东吴证券 - 成交价改进换手率因子

原理:
1. 使用威廉式(Bar Chart)上下影线定义:
   - 上影线 = High - Close (卖压)
   - 下影线 = Close - Low (买压)
2. PLUS = (下影线 - 上影线) / 昨日收盘价
   = (2*Close - High - Low) / Close[t-1]

因子解读:
- PLUS > 0: 下影线 > 上影线，买压大于卖压
- PLUS < 0: 上影线 > 下影线，卖压大于买压
- 研报中 RankIC = -0.06，即 PLUS 值越大，未来收益越低
- 本因子取反，使得高值表示看多信号

应用:
- 日内多空情绪因子
- 可与换手率因子配合使用
"""
import numpy as np
from lumina.impulse.fixed import *


def ki005(high, low, close, window, weriod, ewm=False):
    """
    PLUS 影线差因子

    参数:
        high: 最高价 (1分钟数据)
        low: 最低价 (1分钟数据)
        close: 收盘价 (1分钟数据)
        window: 外层平滑窗口
        weriod: 计算周期 (日内周期)
        ewm: 是否使用指数加权

    返回:
        alpha: PLUS 因子值 (已取反，高值表示看多)
    """
    method = 'ewm' if ewm else 'rolling'

    # 1. 计算昨日收盘价 (前一根K线的收盘价)
    prev_close = close.shift(1)

    # 2. 计算 PLUS = (2*Close - High - Low) / Close[t-1]
    # 等价于 (下影线 - 上影线) / 昨收
    plus_raw = safe_div(2 * close - high - low, prev_close)

    # 3. 日内均值
    plus_intraday = roller_mean(plus_raw, weriod, weriod, method)

    # 4. 取反 (原始因子 RankIC 为负，取反后高值=看多)
    core1 = -plus_intraday

    # 5. 最终用 window 做平滑
    alpha = roller_mean(core1, window, window, method)

    return alpha
