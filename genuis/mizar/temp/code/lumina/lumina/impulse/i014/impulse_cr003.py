from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr003 import cr003 as calc_cr003

class ImpulseCr003(ImpulseBase):
    """
    cr003：N期最高价与最低价的对数比率波动因子，衡量价格区间的波动幅度。
    计算方式：取N期内最高价与最低价的对数之差，做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr003_keys = default_keys

    @property
    def name(self):
        return "cr003"

    def calc_impulse(self, kl_pd):
        """
        cr003：N期最高价与最低价的对数比率波动因子，衡量价格区间的波动幅度。
        """
        impulse_dict = {}
        for dk in self.cr003_keys:
            cr003 = calc_cr003(high=kl_pd['high'], low=kl_pd['low'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr003 = self._format(cr003, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr003
        return impulse_dict 