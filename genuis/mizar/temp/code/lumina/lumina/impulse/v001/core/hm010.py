import pandas as pd
from lumina.impulse.fixed import *

## 计算股票连续登上龙虎榜的天数。天数越多，说明资金关注度持续火热。
## 依赖 on_list 列，并将其 NaN 填充为 0
def hm010(on_list, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    # 核心步骤1: 将NaN填充为0，创建0/1事件序列
    on_list_flag = on_list.fillna(0)
    
    # 核心步骤2: 对DataFrame的每一列（每只股票）分别计算连续天数
    # 初始化一个空的DataFrame来存放结果
    alpha1 = pd.DataFrame(index=on_list_flag.index)
    
    # 遍历每一列（即每只股票）
    for code in on_list_flag.columns:
        # 提取当前股票的Series数据
        series_flag = on_list_flag[code]
        
        # 对这个Series应用之前的逻辑
        consecutive_blocks = (series_flag != series_flag.shift()).cumsum()
        consecutive_days = series_flag.groupby(consecutive_blocks).cumsum()
        
        # 将计算结果存入结果DataFrame
        alpha1[code] = consecutive_days

    # weriod 在这里可以理解为对连续天数的一个截断或调整，但此处我们直接使用原始连续天数
    # window 用于最终平滑
    alpha = roller_mean(alpha1, window, 1, method)
    return alpha