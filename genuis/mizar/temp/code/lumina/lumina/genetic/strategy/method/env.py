import numpy as np

'''
# 默认参数设置
# 默认最大持仓手数
default_volume = 1
# 默认持仓周期范围
default_holding_period = np.arange(5, 31, 5)
# 默认移动止损百分比范围
default_trailing_percent = np.arange(1 / 100, 5 / 100, 0.5 / 100)
# 默认ATR周期范围
default_atr_period = np.arange(5, 31, 5)
# 默认ATR乘数范围
default_atr_multiplier = np.arange(2, 10, 1)
# 默认滚动窗口范围
default_rolling_num = np.arange(5, 31, 5)
# 统一管理的最大持仓手数集合
'''
default_max_volume = [1]


class Function(object):
    """
    策略函数包装类
    用于统一存储策略函数、参数和名称
    """
    def __init__(self, function, params, name):
        self.function = function
        self.params = params
        self.name = name
