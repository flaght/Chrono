from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr029 import cr029 as calc_cr029

class ImpulseCr029(ImpulseBase):
    """
    cr029：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合峰度复合因子，衡量收益、极端波动与量能变化的高阶陡峭性。
    计算方式：先计算N期收盘价对数收益率、N期最高-最低极差、N期成交量变化率的三阶混合峰度，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr029_keys = default_keys

    @property
    def name(self):
        return "cr029"

    def calc_impulse(self, kl_pd):
        """
        cr029：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合峰度复合因子，衡量收益、极端波动与量能变化的高阶陡峭性。
        """
        impulse_dict = {}
        for dk in self.cr029_keys:
            cr029 = calc_cr029(close=kl_pd['close'], high=kl_pd['high'], low=kl_pd['low'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr029 = self._format(cr029, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr029
        return impulse_dict 