from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr011 import cr011 as calc_cr011

class ImpulseCr011(ImpulseBase):
    """
    cr011：N日收盘价与开盘价的对数收益率与成交量变化率的协方差复合因子，衡量价量联动的波动性。
    计算方式：先计算N日收盘-开盘对数收益率与成交量变化率的协方差，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr011_keys = default_keys

    @property
    def name(self):
        return "cr011"

    def calc_impulse(self, kl_pd):
        """
        cr011：N日收盘价与开盘价的对数收益率与成交量变化率的协方差复合因子，衡量价量联动的波动性。
        """
        impulse_dict = {}
        for dk in self.cr011_keys:
            cr011 = calc_cr011(close=kl_pd['close'], open=kl_pd['open'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr011 = self._format(cr011, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr011
        return impulse_dict 