from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr021 import cr021 as calc_cr021

class ImpulseCr021(ImpulseBase):
    """
    cr021：N期收盘价对数收益率的偏度与成交量变化率的相关系数复合因子，衡量收益分布偏斜与量能变化的同步性。
    计算方式：先计算N期收盘价对数收益率的偏度与N期成交量变化率的相关系数，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr021_keys = default_keys

    @property
    def name(self):
        return "cr021"

    def calc_impulse(self, kl_pd):
        """
        cr021：N期收盘价对数收益率的偏度与成交量变化率的相关系数复合因子，衡量收益分布偏斜与量能变化的同步性。
        """
        impulse_dict = {}
        for dk in self.cr021_keys:
            cr021 = calc_cr021(close=kl_pd['close'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr021 = self._format(cr021, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr021
        return impulse_dict 