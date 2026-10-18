from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr060 import cr060 as calc_cr060

class ImpulseCr060(ImpulseBase):
    """
    cr060：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合极端值复合因子，衡量收益、极端波动与持仓变化的高阶极端风险。
    计算方式：先计算N期收盘价对数收益率、N期最高-最低极差、N期持仓量变化率的三阶混合极端值（如最大/最小），再做滑动平均。
    本因子为cr034的持仓量(openint)版本。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr060_keys = default_keys

    @property
    def name(self):
        return "cr060"

    def calc_impulse(self, kl_pd):
        """
        cr060：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合极端值复合因子，衡量收益、极端波动与持仓变化的高阶极端风险。
        """
        impulse_dict = {}
        for dk in self.cr060_keys:
            cr060 = calc_cr060(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], openint=kl_pd['openint'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr060 = self._format(cr060, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr060
        return impulse_dict 