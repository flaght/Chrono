"""
因子: dz003
来源: 东证期货 - 国债期货量价因子挖掘 (2022-07-12)
批次: batch0008
"""
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.ki008 import ki008 as calc_ki008


class ImpulseKi008(ImpulseBase):
    """四分位差信号强度因子"""

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki008_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki008"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki008_keys:
            factor = calc_ki008(
                high=kl_pd['high'],
                low=kl_pd['low'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            factor = self._format(factor, name=name)
            impulse_dict[name] = factor
        return impulse_dict
