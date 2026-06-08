from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr037 import cr037 as calc_cr037

class ImpulseCr037(ImpulseBase):
    """
    cr037：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合移动窗口正负极性因子，衡量高阶极性变化。
    计算方式：三变量中心化后乘积在N期内正负号变化次数，归一化后滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr037_keys = default_keys

    @property
    def name(self):
        return "cr037"

    def calc_impulse(self, kl_pd):
        """
        cr037：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合移动窗口正负极性因子，衡量高阶极性变化。
        """
        impulse_dict = {}
        for dk in self.cr037_keys:
            cr037 = calc_cr037(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr037 = self._format(cr037, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr037
        return impulse_dict 