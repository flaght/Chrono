from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr035 import cr035 as calc_cr035

class ImpulseCr035(ImpulseBase):
    """
    cr035：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合变异系数复合因子，衡量收益、极端波动与量能变化的高阶相对波动性。
    计算方式：先计算N期收盘价对数收益率、N期最高-最低极差、N期成交量变化率的三阶混合变异系数，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr035_keys = default_keys

    @property
    def name(self):
        return "cr035"

    def calc_impulse(self, kl_pd):
        """
        cr035：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合变异系数复合因子，衡量收益、极端波动与量能变化的高阶相对波动性。
        """
        impulse_dict = {}
        for dk in self.cr035_keys:
            cr035 = calc_cr035(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr035 = self._format(cr035, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr035
        return impulse_dict 