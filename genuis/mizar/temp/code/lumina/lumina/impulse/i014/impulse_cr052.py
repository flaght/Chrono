from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr052 import cr052 as calc_cr052

class ImpulseCr052(ImpulseBase):
    """
    cr052：持仓量(openint)与长期均值偏离度的tanh非线性压缩因子，衡量持仓均值回归强度。
    计算方式：持仓量与N期均值之差标准化后，经过tanh变换，滑动平均。
    本因子为cr042的持仓量(openint)版本。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr052_keys = default_keys

    @property
    def name(self):
        return "cr052"

    def calc_impulse(self, kl_pd):
        """
        cr052：持仓量(openint)与长期均值偏离度的tanh非线性压缩因子，衡量持仓均值回归强度。
        """
        impulse_dict = {}
        for dk in self.cr052_keys:
            cr052 = calc_cr052(openint=kl_pd['openint'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr052 = self._format(cr052, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr052
        return impulse_dict 