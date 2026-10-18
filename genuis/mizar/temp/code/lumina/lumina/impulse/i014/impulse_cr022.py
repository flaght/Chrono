from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr022 import cr022 as calc_cr022

class ImpulseCr022(ImpulseBase):
    """
    cr022：N期最高价与最低价极差的偏度与收盘价波动率的协方差复合因子，衡量极端波动分布与风险的联动性。
    计算方式：先计算N期最高-最低极差的偏度与N期收盘价标准差的协方差，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr022_keys = default_keys

    @property
    def name(self):
        return "cr022"

    def calc_impulse(self, kl_pd):
        """
        cr022：N期最高价与最低价极差的偏度与收盘价波动率的协方差复合因子，衡量极端波动分布与风险的联动性。
        """
        impulse_dict = {}
        for dk in self.cr022_keys:
            cr022 = calc_cr022(high=kl_pd['high'], low=kl_pd['low'], close=kl_pd['close'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr022 = self._format(cr022, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr022
        return impulse_dict 