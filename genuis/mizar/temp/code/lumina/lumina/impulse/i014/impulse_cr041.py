from lumina.impulse.base import ImpulseBase, default_keys2
from lumina.impulse.i014.core.cr041 import cr041 as calc_cr041


class ImpulseCr041(ImpulseBase):
    """
    cr041：短期与长期波动率比值的sigmoid变换因子，衡量波动率聚集与极端变化。
    计算方式：N1期与N2期收盘价对数收益率标准差之比，经过sigmoid变换后滑动平均。
    """

    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.cr041_keys = default_keys

    @property
    def name(self):
        return "cr041"

    def calc_impulse(self, kl_pd):
        """
        cr041：短期与长期波动率比值的sigmoid变换因子，衡量波动率聚集与极端变化。
        """
        impulse_dict = {}
        for dk in self.cr041_keys:
            cr041 = calc_cr041(close=kl_pd['close'],
                               window=dk[0],
                               fast=dk[1],
                               slow=dk[2],
                               ewm=True if dk[3] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}_{dk[3]}"
            cr041 = self._format(cr041, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr041
        return impulse_dict
