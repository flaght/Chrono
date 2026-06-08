from lumina.impulse.base import ImpulseBase, default_keys1
from lumina.impulse.i014.core.cr007 import cr007 as calc_cr007

class ImpulseCr007(ImpulseBase):
    """
    cr007：N日收盘价动量与波动率复合因子，衡量价格趋势与风险的综合效应。
    计算方式：先计算N日动量（收盘价/前N日收盘价-1），再计算N日收益率标准差，两者归一化后相乘，最后做滑动平均。
    """
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cr007_keys = default_keys

    @property
    def name(self):
        return "cr007"

    def calc_impulse(self, kl_pd):
        """
        cr007：N日收盘价动量与波动率复合因子，衡量价格趋势与风险的综合效应。
        """
        impulse_dict = {}
        for dk in self.cr007_keys:
            cr007 = calc_cr007(close=kl_pd['close'], window=dk[0], weriod=dk[1], ewm=True if dk[2] == 1 else False)
            name = f"{self.name}_{dk[0]}_{dk[1]}_{dk[2]}"
            cr007 = self._format(cr007, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = cr007
        return impulse_dict 