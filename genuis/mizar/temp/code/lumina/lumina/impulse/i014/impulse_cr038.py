from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr038 import cr038 as calc_cr038

class ImpulseCr038(ImpulseBase):
    """
    cr038：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合移动窗口零穿越因子，衡量高阶零点穿越频率。
    计算方式：三变量中心化后乘积在N期内穿越零点的次数，归一化后滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr038_keys = default_keys

    @property
    def name(self):
        return "cr038"

    def calc_impulse(self, kl_pd):
        """
        cr038：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合移动窗口零穿越因子，衡量高阶零点穿越频率。
        """
        impulse_dict = {}
        for dk in self.cr038_keys:
            cr038 = calc_cr038(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr038 = self._format(cr038, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr038
        return impulse_dict 