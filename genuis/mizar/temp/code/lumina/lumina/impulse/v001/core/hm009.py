import pandas as pd
from lumina.impulse.fixed import *


## 过去N个交易日中，登上龙虎榜的总天数，衡量个股在近期被游资关注的频繁程度
## 依赖 on_list 列，并将其 NaN 填充为 0
def hm009(on_list, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    # 核心步骤1: 将NaN填充为0，创建0/1事件序列
    on_list_flag = on_list.fillna(0)

    # 核心步骤2: 在 weriod 窗口内滚动求和，得到上榜频率
    # 假设存在 roller_sum 函数，与 roller_mean 类似
    alpha1 = roller_sum(on_list_flag, weriod, 1, method)

    # 最终步骤: 对结果进行平滑处理
    alpha = roller_mean(alpha1, window, 1, method)
    return alpha
