from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr028 import cr028 as calc_cr028

class ImpulseCr028(ImpulseBase):
    """
    cr028：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合偏度复合因子，衡量收益、极端波动与量能变化的高阶非对称性。
    计算方式：先计算N期收盘价对数收益率、N期最高-最低极差、N期成交量变化率的三阶混合偏度，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr028_keys = default_keys

    @property
    def name(self):
        return "cr028"

    def calc_impulse(self, kl_pd):
        """
        cr028：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合偏度复合因子，衡量收益、极端波动与量能变化的高阶非对称性。
        """
        impulse_dict = {}
        for dk in self.cr028_keys:
            cr028 = calc_cr028(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr028 = self._format(cr028, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr028
        return impulse_dict 