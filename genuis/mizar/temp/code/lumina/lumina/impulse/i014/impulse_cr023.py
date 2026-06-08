from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr023 import cr023 as calc_cr023

class ImpulseCr023(ImpulseBase):
    """
    cr023：N期收盘价对数收益率的峰度与成交量波动率的相关系数复合因子，衡量收益分布陡峭与量能风险的同步性。
    计算方式：先计算N期收盘价对数收益率的峰度与N期成交量标准差的相关系数，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr023_keys = default_keys

    @property
    def name(self):
        return "cr023"

    def calc_impulse(self, kl_pd):
        """
        cr023：N期收盘价对数收益率的峰度与成交量波动率的相关系数复合因子，衡量收益分布陡峭与量能风险的同步性。
        """
        impulse_dict = {}
        for dk in self.cr023_keys:
            cr023 = calc_cr023(close=kl_pd['close'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr023 = self._format(cr023, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr023
        return impulse_dict 