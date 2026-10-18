# -*- encoding:utf-8 -*-
from lumina.position.base import PositionBase
import pdb
"""
    默认0.01即1% atr.g_atr_pos_base = 0.01修改仓位基础配比
    需要注意外部其它自定义仓位管理类不要随意使用模块全局变量，AtrPosition特殊因为注册
    在EnvProcess中在多进程启动时拷贝了模块全局设置内存
"""

g_atr_pos_base = 0.01


class AtrPosition(PositionBase):

    s_atr_base_price = 15  # best fit wide: 12-20
    s_std_atr_threshold = 0.5  # best fit wide: 0.3-0.65

    def fit_weight(self, factor_object):
        """
        fit_position计算的结果是买入多少个单位（股，手，顿，合约）
        计算：（常数价格 ／ 买入价格）＊ 当天交易日atr21
        :param factor_object: ABuFactorBuyBases实例对象
        :return: read_cash有具体值则返回买入多少个单位（股，手，顿，合约），否则返回买入比例
        """
        std_atr = (self.s_atr_base_price / self.bp) * self.kl_pd_buy['atr21']
        """
            对atr 进行限制 避免由于股价波动过小，导致
            atr小，产生大量买单，实际上针对这种波动异常（过小，过大）的股票
            需要有其它的筛选过滤策略, 选股的时候取0.5，这样最大取两倍g_atr_pos_base
        """
        atr_wv = self.std_atr_threshold if std_atr < self.std_atr_threshold else std_atr
        # 计算出仓位比例
        atr_pos = self.atr_pos_base / atr_wv
        # 最大仓位限制
        atr_pos = self.pos_max if atr_pos > self.pos_max else atr_pos
        return atr_pos

    def fit_position(self, factor_object):
        pos = self.fit_weight(factor_object)
        return self.read_cash * pos / self.bp * self.deposit_rate

    def _init_self(self, **kwargs):
        """atr仓位控制管理类初始化设置"""
        self.atr_base_price = kwargs.pop('atr_base_price',
                                         AtrPosition.s_atr_base_price)
        self.std_atr_threshold = kwargs.pop('std_atr_threshold',
                                            AtrPosition.s_std_atr_threshold)
        self.atr_pos_base = kwargs.pop('atr_pos_base', g_atr_pos_base)