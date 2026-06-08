from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr009 import cr009 as calc_cr009

class ImpulseCr009(ImpulseBase):
    """
    cr009：N日收盘价与成交量的相关性与极端值复合因子，衡量价量共振与极端波动。
    计算方式：先计算N日收盘价与成交量的相关系数，再与N日收盘价的极端涨跌幅相乘，最后做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr009_keys = default_keys

    @property
    def name(self):
        return "cr009"

    def calc_impulse(self, kl_pd):
        """
        cr009：N日收盘价与成交量的相关性与极端值复合因子，衡量价量共振与极端波动。
        """
        impulse_dict = {}
        for dk in self.cr009_keys:
            cr009 = calc_cr009(close=kl_pd['close'], volume=kl_pd['volume'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr009 = self._format(cr009, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr009
        return impulse_dict 