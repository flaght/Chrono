from lumina.impulse.base import ImpulseBase, default_keys3
from lumina.impulse.i014.core.cr050 import cr050 as calc_cr050


class ImpulseCr050(ImpulseBase):
    """
    cr050：多窗口持仓量(openint)极差比值的滑动窗口排序分位因子，衡量多尺度持仓极端波动。
    计算方式：短期持仓量极差与长期持仓量极差之比，在N期内排序分位，滑动平均。
    本因子为cr045的持仓量(openint)版本。
    """

    def __init__(self, **kwargs):
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.cr050_keys = default_keys

    @property
    def name(self):
        return "cr050"

    def calc_impulse(self, kl_pd):
        """
        cr050：多窗口持仓量(openint)极差比值的滑动窗口排序分位因子，衡量多尺度持仓极端波动。
        """
        impulse_dict = {}
        for dk in self.cr050_keys:
            cr050 = calc_cr050(openint=kl_pd['openint'],
                               slow=dk[0],
                               fast=dk[1],
                               weriod=dk[2],
                               window=dk[3],
                               ewm=True if dk[4] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}_{dk[3]}_{dk[4]}"
            cr050 = self._format(cr050, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr050
        return impulse_dict
