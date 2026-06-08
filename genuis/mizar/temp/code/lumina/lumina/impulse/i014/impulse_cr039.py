from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr039 import cr039 as calc_cr039

class ImpulseCr039(ImpulseBase):
    """
    cr039：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合移动窗口绝对值均值因子，衡量高阶绝对波动性。
    计算方式：三变量中心化后乘积在N期内绝对值均值，滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr039_keys = default_keys

    @property
    def name(self):
        return "cr039"

    def calc_impulse(self, kl_pd):
        """
        cr039：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合移动窗口绝对值均值因子，衡量高阶绝对波动性。
        """
        impulse_dict = {}
        for dk in self.cr039_keys:
            cr039 = calc_cr039(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr039 = self._format(cr039, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr039
        return impulse_dict 