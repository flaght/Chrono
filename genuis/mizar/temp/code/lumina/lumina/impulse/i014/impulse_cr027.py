from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr027 import cr027 as calc_cr027

class ImpulseCr027(ImpulseBase):
    """
    cr027：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合协方差复合因子，衡量收益、极端波动与量能变化的高阶联动性。
    计算方式：先计算N期收盘价对数收益率、N期最高-最低极差、N期成交量变化率的三阶混合协方差，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr027_keys = default_keys

    @property
    def name(self):
        return "cr027"

    def calc_impulse(self, kl_pd):
        """
        cr027：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合协方差复合因子，衡量收益、极端波动与量能变化的高阶联动性。
        """
        impulse_dict = {}
        for dk in self.cr027_keys:
            cr027 = calc_cr027(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr027 = self._format(cr027, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr027
        return impulse_dict 