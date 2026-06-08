from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr012 import cr012 as calc_cr012

class ImpulseCr012(ImpulseBase):
    """
    cr012：N日最高价与最低价的极差与收盘价波动率复合因子，衡量极端波动与风险。
    计算方式：先计算N日最高-最低极差，再与N日收盘价标准差相乘，最后做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr012_keys = default_keys

    @property
    def name(self):
        return "cr012"

    def calc_impulse(self, kl_pd):
        """
        cr012：N日最高价与最低价的极差与收盘价波动率复合因子，衡量极端波动与风险。
        """
        impulse_dict = {}
        for dk in self.cr012_keys:
            cr012 = calc_cr012(high=kl_pd['high'], low=kl_pd['low'], close=kl_pd['close'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr012 = self._format(cr012, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr012
        return impulse_dict 