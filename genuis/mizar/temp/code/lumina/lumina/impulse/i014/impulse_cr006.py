from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr006 import cr006 as calc_cr006

class ImpulseCr006(ImpulseBase):
    """
    cr006：N日最高价与收盘价的相关系数因子，衡量价格高点与收盘的同步性。
    计算方式：计算N日内最高价与收盘价的相关系数，做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr006_keys = default_keys

    @property
    def name(self):
        return "cr006"

    def calc_impulse(self, kl_pd):
        """
        cr006：N日最高价与收盘价的相关系数因子，衡量价格高点与收盘的同步性。
        """
        impulse_dict = {}
        for dk in self.cr006_keys:
            cr006 = calc_cr006(high=kl_pd['high'], close=kl_pd['close'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr006 = self._format(cr006, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr006
        return impulse_dict 