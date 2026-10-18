from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr042 import cr042 as calc_cr042

class ImpulseCr042(ImpulseBase):
    """
    cr042：收盘价与长期均值偏离度的tanh非线性压缩因子，衡量均值回归强度。
    计算方式：收盘价与N期均值之差标准化后，经过tanh变换，滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr042_keys = default_keys

    @property
    def name(self):
        return "cr042"

    def calc_impulse(self, kl_pd):
        """
        cr042：收盘价与长期均值偏离度的tanh非线性压缩因子，衡量均值回归强度。
        """
        impulse_dict = {}
        for dk in self.cr042_keys:
            cr042 = calc_cr042(close=kl_pd['close'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr042 = self._format(cr042, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr042
        return impulse_dict 