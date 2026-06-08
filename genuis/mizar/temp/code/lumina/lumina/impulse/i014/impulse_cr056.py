from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr056 import cr056 as calc_cr056

class ImpulseCr056(ImpulseBase):
    """
    cr056：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合移动窗口零穿越因子，衡量高阶持仓零点穿越频率。
    计算方式：三变量中心化后乘积在N期内穿越零点的次数，归一化后滑动平均。
    本因子为cr038的持仓量(openint)版本。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr056_keys = default_keys

    @property
    def name(self):
        return "cr056"

    def calc_impulse(self, kl_pd):
        """
        cr056：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合移动窗口零穿越因子，衡量高阶持仓零点穿越频率。
        """
        impulse_dict = {}
        for dk in self.cr056_keys:
            cr056 = calc_cr056(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], openint=kl_pd['openint'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr056 = self._format(cr056, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr056
        return impulse_dict 