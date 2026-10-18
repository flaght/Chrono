from lumina.impulse.base import ImpulseBase, default_keys3
from lumina.impulse.i014.core.cr045 import cr045 as calc_cr045


class ImpulseCr045(ImpulseBase):
    """
    cr045：多窗口极差比值的滑动窗口排序分位因子，衡量多尺度极端波动。
    计算方式：短期极差与长期极差之比，在N期内排序分位，滑动平均。
    """

    def __init__(self, **kwargs):
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.cr045_keys = default_keys

    @property
    def name(self):
        return "cr045"

    def calc_impulse(self, kl_pd):
        """
        cr045：多窗口极差比值的滑动窗口排序分位因子，衡量多尺度极端波动。
        """
        impulse_dict = {}
        for dk in self.cr045_keys:
            cr045 = calc_cr045(high=kl_pd['high'],
                               low=kl_pd['low'],
                               window=dk[0],
                               fast=dk[1],
                               slow=dk[2],
                               weriod=dk[3],
                               ewm=True if dk[4] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}_{dk[3]}_{dk[4]}"
            cr045 = self._format(cr045, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr045
        return impulse_dict
