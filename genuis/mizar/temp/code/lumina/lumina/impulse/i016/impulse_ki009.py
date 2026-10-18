# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.ki009 import ki009 as calc_ki009

class ImpulseKi009(ImpulseBase):
    """
    Dual Thrust 双向突破因子

    基于商品期货CTA策略的经典突破系统，通过计算价格通道来捕捉突破信号。
    适用于趋势性行情的捕捉，对突破上轨做多、突破下轨做空。
    """

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki009_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki009_keys:
            factor = calc_ki009(
                open=kl_pd['open'],
                high=kl_pd['high'],
                low=kl_pd['low'],
                close=kl_pd['close'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            factor = self._format(factor, name=name)
            impulse_dict[name] = factor
        return impulse_dict
