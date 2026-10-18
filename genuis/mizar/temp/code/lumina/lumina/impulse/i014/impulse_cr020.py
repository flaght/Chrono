from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr020 import cr020 as calc_cr020

class ImpulseCr020(ImpulseBase):
    """
    cr020：N期最高价与最低价极差与成交量变化率的协方差复合因子，衡量极端波动与量能变化的联动性。
    计算方式：先计算N期最高-最低极差与N期成交量变化率的协方差，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr020_keys = default_keys

    @property
    def name(self):
        return "cr020"

    def calc_impulse(self, kl_pd):
        """
        cr020：N期最高价与最低价极差与成交量变化率的协方差复合因子，衡量极端波动与量能变化的联动性。
        """
        impulse_dict = {}
        for dk in self.cr020_keys:
            cr020 = calc_cr020(high=kl_pd['high'], low=kl_pd['low'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr020 = self._format(cr020, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr020
        return impulse_dict 