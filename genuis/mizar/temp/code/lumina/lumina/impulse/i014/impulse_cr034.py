from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr034 import cr034 as calc_cr034

class ImpulseCr034(ImpulseBase):
    """
    cr034：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合极端值复合因子，衡量收益、极端波动与量能变化的高阶极端风险。
    计算方式：先计算N期收盘价对数收益率、N期最高-最低极差、N期成交量变化率的三阶混合极端值（如最大/最小），再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr034_keys = default_keys

    @property
    def name(self):
        return "cr034"

    def calc_impulse(self, kl_pd):
        """
        cr034：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合极端值复合因子，衡量收益、极端波动与量能变化的高阶极端风险。
        """
        impulse_dict = {}
        for dk in self.cr034_keys:
            cr034 = calc_cr034(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr034 = self._format(cr034, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr034
        return impulse_dict 