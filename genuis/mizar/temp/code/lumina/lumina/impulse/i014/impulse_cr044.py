from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr044 import cr044 as calc_cr044


class ImpulseCr044(ImpulseBase):
    """
    cr044：高低价极差的自适应分位阈值触发因子，衡量极端行情爆发概率。
    计算方式：N期极差大于N期分位阈值时输出1，否则为0，滑动平均。
    """

    def __init__(self, **kwargs):
        default_keys = [(0.8, 5, 10, 1), (0.8, 5, 10, 0), (0.8, 10, 15, 1),
                        (0.8, 10, 15, 0)] if not kwargs else kwargs.get('keys')
        self.cr044_keys = default_keys

    @property
    def name(self):
        return "cr044"

    def calc_impulse(self, kl_pd):
        """
        cr044：高低价极差的自适应分位阈值触发因子，衡量极端行情爆发概率。
        """
        impulse_dict = {}
        for dk in self.cr044_keys:
            cr044 = calc_cr044(high=kl_pd['high'],
                               low=kl_pd['low'],
                               threshold=dk[0],
                               window=dk[1],
                               weriod=dk[2],
                               ewm=True if dk[3] == 1 else False)
            name = f"{self.name}_{int(dk[0] * 100)}_{dk[1]}_{dk[2]}_{dk[3]}"
            cr044 = self._format(cr044, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr044
        return impulse_dict
