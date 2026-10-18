from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr047 import cr047 as calc_cr047

class ImpulseCr047(ImpulseBase):
    """
    cr047：N期收盘价与持仓量(openint)的相关性与极端值复合因子，衡量价持仓共振与极端波动。
    计算方式：先计算N期收盘价与持仓量(openint)的相关系数，再与N期收盘价的极端涨跌幅相乘，最后做滑动平均。
    本因子为cr009的持仓量(openint)版本。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr047_keys = default_keys

    @property
    def name(self):
        return "cr047"

    def calc_impulse(self, kl_pd):
        """
        cr047：N期收盘价与持仓量(openint)的相关性与极端值复合因子，衡量价持仓共振与极端波动。
        """
        impulse_dict = {}
        for dk in self.cr047_keys:
            cr047 = calc_cr047(close=kl_pd['close'], openint=kl_pd['openint'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr047 = self._format(cr047, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr047
        return impulse_dict 