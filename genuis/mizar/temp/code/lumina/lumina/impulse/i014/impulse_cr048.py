from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr048 import cr048 as calc_cr048

class ImpulseCr048(ImpulseBase):
    """
    cr048：N期收盘价与开盘价的对数收益率偏度与持仓量(openint)波动率复合因子，衡量收益分布偏斜与持仓风险。
    计算方式：先计算N期收盘-开盘对数收益率的偏度，再与N期持仓量(openint)标准差相乘，最后做滑动平均。
    本因子为cr013的持仓量(openint)版本。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr048_keys = default_keys

    @property
    def name(self):
        return "cr048"

    def calc_impulse(self, kl_pd):
        """
        cr048：N期收盘价与开盘价的对数收益率偏度与持仓量(openint)波动率复合因子，衡量收益分布偏斜与持仓风险。
        """
        impulse_dict = {}
        for dk in self.cr048_keys:
            cr048 = calc_cr048(close=kl_pd['close'], open=kl_pd['open'], openint=kl_pd['openint'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr048 = self._format(cr048, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr048
        return impulse_dict 