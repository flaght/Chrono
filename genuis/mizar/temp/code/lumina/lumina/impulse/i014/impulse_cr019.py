from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr019 import cr019 as calc_cr019

class ImpulseCr019(ImpulseBase):
    """
    cr019：N期收盘价与开盘价的对数收益率与成交量波动率的相关系数复合因子，衡量收益与量能风险的同步性。
    计算方式：先计算N期收盘-开盘对数收益率与N期成交量标准差的相关系数，再做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr019_keys = default_keys

    @property
    def name(self):
        return "cr019"

    def calc_impulse(self, kl_pd):
        """
        cr019：N期收盘价与开盘价的对数收益率与成交量波动率的相关系数复合因子，衡量收益与量能风险的同步性。
        """
        impulse_dict = {}
        for dk in self.cr019_keys:
            cr019 = calc_cr019(close=kl_pd['close'], open=kl_pd['open'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr019 = self._format(cr019, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr019
        return impulse_dict 