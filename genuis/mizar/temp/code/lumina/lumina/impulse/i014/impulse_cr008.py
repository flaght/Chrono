from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr008 import cr008 as calc_cr008

class ImpulseCr008(ImpulseBase):
    """
    cr008：N日最高价与最低价区间突破与成交量变化复合因子，衡量价格突破与量能配合。
    计算方式：先计算N日最高价与最低价的区间突破幅度，再与N日成交量变化率相乘，最后做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr008_keys = default_keys

    @property
    def name(self):
        return "cr008"

    def calc_impulse(self, kl_pd):
        """
        cr008：N日最高价与最低价区间突破与成交量变化复合因子，衡量价格突破与量能配合。
        """
        impulse_dict = {}
        for dk in self.cr008_keys:
            cr008 = calc_cr008(high=kl_pd['high'], low=kl_pd['low'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr008 = self._format(cr008, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr008
        return impulse_dict 