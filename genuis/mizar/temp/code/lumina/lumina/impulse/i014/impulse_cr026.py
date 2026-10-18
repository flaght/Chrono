from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr026 import cr026 as calc_cr026

class ImpulseCr026(ImpulseBase):
    """
    cr026：N期最高价与最低价极差的自相关系数与收盘价波动率的协方差复合因子，衡量极端波动惯性与风险的联动性。
    计算方式：先计算N期最高-最低极差的自相关系数与N期收盘价标准差的协方差，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr026_keys = default_keys

    @property
    def name(self):
        return "cr026"

    def calc_impulse(self, kl_pd):
        """
        cr026：N期最高价与最低价极差的自相关系数与收盘价波动率的协方差复合因子，衡量极端波动惯性与风险的联动性。
        """
        impulse_dict = {}
        for dk in self.cr026_keys:
            cr026 = calc_cr026(high=kl_pd['high'], low=kl_pd['low'], close=kl_pd['close'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr026 = self._format(cr026, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr026
        return impulse_dict 