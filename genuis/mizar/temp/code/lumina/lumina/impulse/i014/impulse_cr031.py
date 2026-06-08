from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr031 import cr031 as calc_cr031


class ImpulseCr031(ImpulseBase):
    """
    cr031：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合分位数复合因子，衡量收益、极端波动与量能变化的高阶极端分布。
    计算方式：先计算N期收盘价对数收益率、N期最高-最低极差、N期成交量变化率的三阶混合分位数（如95%），再做滑动平均。
    """

    def __init__(self, **kwargs):
        default_keys = [(0.9, 5, 10, 1), (0.9, 10, 15, 1), (0.9, 5, 10, 0),
                        (0.9, 10, 15, 0)] if not kwargs else kwargs.get('keys')
        self.cr031_keys = default_keys

    @property
    def name(self):
        return "cr031"

    def calc_impulse(self, kl_pd):
        """
        cr031：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合分位数复合因子，衡量收益、极端波动与量能变化的高阶极端分布。
        """
        impulse_dict = {}
        for dk in self.cr031_keys:
            cr031 = calc_cr031(close=kl_pd['close'],
                               high=kl_pd['high'],
                               low=kl_pd['low'],
                               volume=kl_pd['volume'],
                               threshold=dk[0],
                               window=dk[1],
                               weriod=dk[2],
                               ewm=True if dk[3] == 1 else False)
            name = f"{self.name}_{int(dk[0]*100)}_{dk[0]}_{dk[1]}_{dk[2]}_{dk[3]}"
            cr031 = self._format(cr031, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr031
        return impulse_dict
