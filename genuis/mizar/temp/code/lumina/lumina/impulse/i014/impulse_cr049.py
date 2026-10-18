from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr049 import cr049 as calc_cr049

class ImpulseCr049(ImpulseBase):
    """
    cr049：N期收盘价与开盘价的对数收益率与持仓量(openint)波动率的相关系数复合因子，衡量收益与持仓风险的同步性。
    计算方式：先计算N期收盘-开盘对数收益率与N期持仓量(openint)标准差的相关系数，再做滑动平均。
    本因子为cr019的持仓量(openint)版本。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr049_keys = default_keys

    @property
    def name(self):
        return "cr049"

    def calc_impulse(self, kl_pd):
        """
        cr049：N期收盘价与开盘价的对数收益率与持仓量(openint)波动率的相关系数复合因子，衡量收益与持仓风险的同步性。
        """
        impulse_dict = {}
        for dk in self.cr049_keys:
            cr049 = calc_cr049(close=kl_pd['close'], open=kl_pd['open'], openint=kl_pd['openint'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr049 = self._format(cr049, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr049
        return impulse_dict 