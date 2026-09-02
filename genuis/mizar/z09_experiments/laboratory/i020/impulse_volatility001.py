# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.volatility001 import volatility001 as calc_volatility001


class ImpulseVolatility001(ImpulseBase):

    def __init__(self, **kwargs):
        # 强制：绝不使用 super().__init__(**kwargs)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.volatility001_keys = default_keys

    @property
    def name(self):
        return "volatility001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.volatility001_keys:
            # 只做参数解包，绝无因子业务逻辑
            vol001 = calc_volatility001(close=kl_pd['close'],
                                        window=dk[0],
                                        weriod=dk[1],
                                        ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            vol001 = self._format(vol001, name=name)
            impulse_dict[name] = vol001
        return impulse_dict