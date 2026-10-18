from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr054 import cr054 as calc_cr054

class ImpulseCr054(ImpulseBase):
    """
    cr054：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合移动窗口绝对值中位数因子，衡量高阶持仓绝对中位波动性。
    计算方式：三变量中心化后乘积在N期内绝对值中位数，滑动平均。
    本因子为cr040的持仓量(openint)版本。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr054_keys = default_keys

    @property
    def name(self):
        return "cr054"

    def calc_impulse(self, kl_pd):
        """
        cr054：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合移动窗口绝对值中位数因子，衡量高阶持仓绝对中位波动性。
        """
        impulse_dict = {}
        for dk in self.cr054_keys:
            cr054 = calc_cr054(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], openint=kl_pd['openint'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr054 = self._format(cr054, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr054
        return impulse_dict 