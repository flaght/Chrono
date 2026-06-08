from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr025 import cr025 as calc_cr025

class ImpulseCr025(ImpulseBase):
    """
    cr025：N期收盘价对数收益率的自相关系数与成交量变化率的相关系数复合因子，衡量收益惯性与量能变化的同步性。
    计算方式：先计算N期收盘价对数收益率的自相关系数与N期成交量变化率的相关系数，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr025_keys = default_keys

    @property
    def name(self):
        return "cr025"

    def calc_impulse(self, kl_pd):
        """
        cr025：N期收盘价对数收益率的自相关系数与成交量变化率的相关系数复合因子，衡量收益惯性与量能变化的同步性。
        """
        impulse_dict = {}
        for dk in self.cr025_keys:
            cr025 = calc_cr025(close=kl_pd['close'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr025 = self._format(cr025, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr025
        return impulse_dict 