from lumina.impulse.base import ImpulseBase, default_keys2
from lumina.impulse.i014.core.cr053 import cr053 as calc_cr053

class ImpulseCr053(ImpulseBase):
    """
    cr053：短期与长期持仓量(openint)波动率比值的sigmoid变换因子，衡量持仓波动率聚集与极端变化。
    计算方式：N1期与N2期持仓量标准差之比，经过sigmoid变换后滑动平均。
    本因子为cr041的持仓量(openint)版本。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.cr053_keys = default_keys

    @property
    def name(self):
        return "cr053"

    def calc_impulse(self, kl_pd):
        """
        cr053：短期与长期持仓量(openint)波动率比值的sigmoid变换因子，衡量持仓波动率聚集与极端变化。
        """
        impulse_dict = {}
        for dk in self.cr053_keys:
            cr053 = calc_cr053(openint=kl_pd['openint'], slow=dk[0], fast=dk[1], window=dk[2], ewm=True if dk[3] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}_{dk[3]}"
            cr053 = self._format(cr053, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr053
        return impulse_dict 