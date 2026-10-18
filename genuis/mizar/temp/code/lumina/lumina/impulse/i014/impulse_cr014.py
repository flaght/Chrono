from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr014 import cr014 as calc_cr014

class ImpulseCr014(ImpulseBase):
    """
    cr014：N日收盘价收益率的偏度与最高价极差复合因子，衡量收益分布偏斜与高点波动。
    计算方式：先计算N日收盘价对数收益率的偏度，再与N日最高价极差相乘，最后做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr014_keys = default_keys

    @property
    def name(self):
        return "cr014"

    def calc_impulse(self, kl_pd):
        """
        cr014：N日收盘价收益率的偏度与最高价极差复合因子，衡量收益分布偏斜与高点波动。
        """
        impulse_dict = {}
        for dk in self.cr014_keys:
            cr014 = calc_cr014(close=kl_pd['close'], high=kl_pd['high'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr014 = self._format(cr014, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr014
        return impulse_dict 