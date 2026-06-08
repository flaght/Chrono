from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr055 import cr055 as calc_cr055

class ImpulseCr055(ImpulseBase):
    """
    cr055：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合移动窗口绝对值均值因子，衡量高阶持仓绝对波动性。
    计算方式：三变量中心化后乘积在N期内绝对值均值，滑动平均。
    本因子为cr039的持仓量(openint)版本。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr055_keys = default_keys

    @property
    def name(self):
        return "cr055"

    def calc_impulse(self, kl_pd):
        """
        cr055：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合移动窗口绝对值均值因子，衡量高阶持仓绝对波动性。
        """
        impulse_dict = {}
        for dk in self.cr055_keys:
            cr055 = calc_cr055(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], openint=kl_pd['openint'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr055 = self._format(cr055, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr055
        return impulse_dict 