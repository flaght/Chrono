from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr046 import cr046 as calc_cr046

class ImpulseCr046(ImpulseBase):
    """
    cr046：N期最高价与最低价区间突破与持仓量(openint)变化复合因子，衡量价格突破与持仓量配合。
    计算方式：先计算N期最高价与最低价的区间突破幅度，再与N期持仓量(openint)变化率相乘，最后做滑动平均。
    本因子为cr008的持仓量(openint)版本。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr046_keys = default_keys

    @property
    def name(self):
        return "cr046"

    def calc_impulse(self, kl_pd):
        """
        cr046：N期最高价与最低价区间突破与持仓量(openint)变化复合因子，衡量价格突破与持仓量配合。
        """
        impulse_dict = {}
        for dk in self.cr046_keys:
            cr046 = calc_cr046(high=kl_pd['high'], low=kl_pd['low'], openint=kl_pd['openint'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr046 = self._format(cr046, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr046
        return impulse_dict 