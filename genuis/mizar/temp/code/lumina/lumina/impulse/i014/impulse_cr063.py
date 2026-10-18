from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr063 import cr063 as calc_cr063


class ImpulseCr063(ImpulseBase):
    """
    cr063：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合分位数复合因子，衡量收益、极端波动与持仓变化的高阶极端分布。
    计算方式：先计算N期收盘价对数收益率、N期最高-最低极差、N期持仓量变化率的三阶混合分位数（如95%），再做滑动平均。
    本因子为cr031的持仓量(openint)版本。
    """

    def __init__(self, **kwargs):
        default_keys = [(0.9, 5, 10, 1), (0.9, 10, 15, 1), (0.9, 5, 10, 0),
                        (0.9, 10, 15, 0)] if not kwargs else kwargs.get('keys')
        self.cr063_keys = default_keys

    @property
    def name(self):
        return "cr063"

    def calc_impulse(self, kl_pd):
        """
        cr063：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合分位数复合因子，衡量收益、极端波动与持仓变化的高阶极端分布。
        """
        impulse_dict = {}
        for dk in self.cr063_keys:
            cr063 = calc_cr063(close=kl_pd['close'],
                               high=kl_pd['high'],
                               low=kl_pd['low'],
                               openint=kl_pd['openint'],
                               threshold=dk[0],
                               window=dk[1],
                               weriod=dk[2],
                               ewm=True if dk[3] == 1 else False)
            name = f"{self.name}_{int(dk[0]*100)}_{dk[0]}_{dk[1]}_{dk[2]}_{dk[3]}"
            cr063 = self._format(cr063, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr063
        return impulse_dict
