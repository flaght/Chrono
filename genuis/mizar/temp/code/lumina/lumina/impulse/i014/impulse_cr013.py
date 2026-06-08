from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr013 import cr013 as calc_cr013

class ImpulseCr013(ImpulseBase):
    """
    cr013：N日收盘价与开盘价的对数收益率偏度与成交量波动率复合因子，衡量收益分布偏斜与量能风险。
    计算方式：先计算N日收盘-开盘对数收益率的偏度，再与N日成交量标准差相乘，最后做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr013_keys = default_keys

    @property
    def name(self):
        return "cr013"

    def calc_impulse(self, kl_pd):
        """
        cr013：N日收盘价与开盘价的对数收益率偏度与成交量波动率复合因子，衡量收益分布偏斜与量能风险。
        """
        impulse_dict = {}
        for dk in self.cr013_keys:
            cr013 = calc_cr013(close=kl_pd['close'], open=kl_pd['open'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr013 = self._format(cr013, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr013
        return impulse_dict 