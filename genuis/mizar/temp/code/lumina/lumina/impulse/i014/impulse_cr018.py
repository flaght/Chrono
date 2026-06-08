from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr018 import cr018 as calc_cr018

class ImpulseCr018(ImpulseBase):
    """
    cr018：N日收盘价与开盘价的对数收益率与最低价极差的相关系数复合因子，衡量收益与低点波动的同步性。
    计算方式：先计算N日收盘-开盘对数收益率与最低价极差的相关系数，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr018_keys = default_keys

    @property
    def name(self):
        return "cr018"

    def calc_impulse(self, kl_pd):
        """
        cr018：N日收盘价与开盘价的对数收益率与最低价极差的相关系数复合因子，衡量收益与低点波动的同步性。
        """
        impulse_dict = {}
        for dk in self.cr018_keys:
            cr018 = calc_cr018(close=kl_pd['close'], open=kl_pd['open'], low=kl_pd['low'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr018 = self._format(cr018, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr018
        return impulse_dict 