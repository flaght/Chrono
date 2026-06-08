from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr015 import cr015 as calc_cr015

class ImpulseCr015(ImpulseBase):
    """
    cr015：N日收盘价与开盘价的对数收益率与最高价极差的相关系数复合因子，衡量收益与高点波动的同步性。
    计算方式：先计算N日收盘-开盘对数收益率与最高价极差的相关系数，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr015_keys = default_keys

    @property
    def name(self):
        return "cr015"

    def calc_impulse(self, kl_pd):
        """
        cr015：N日收盘价与开盘价的对数收益率与最高价极差的相关系数复合因子，衡量收益与高点波动的同步性。
        """
        impulse_dict = {}
        for dk in self.cr015_keys:
            cr015 = calc_cr015(close=kl_pd['close'], open=kl_pd['open'], high=kl_pd['high'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr015 = self._format(cr015, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr015
        return impulse_dict 