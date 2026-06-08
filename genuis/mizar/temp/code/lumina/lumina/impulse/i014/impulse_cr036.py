from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr036 import cr036 as calc_cr036

class ImpulseCr036(ImpulseBase):
    """
    cr036：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合移动窗口排序分位因子，衡量高阶排序极端性。
    计算方式：三变量中心化后乘积在N期内排序，取分位排名（如90%），再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr036_keys = default_keys

    @property
    def name(self):
        return "cr036"

    def calc_impulse(self, kl_pd):
        """
        cr036：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合移动窗口排序分位因子，衡量高阶排序极端性。
        """
        impulse_dict = {}
        for dk in self.cr036_keys:
            cr036 = calc_cr036(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr036 = self._format(cr036, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr036
        return impulse_dict 