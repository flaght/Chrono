# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys0
from laboratory.i020.core.ybf001 import ybf001 as calc_ybf001


class ImpulseYbf001(ImpulseBase):

    def __init__(self, **kwargs):
        # 强制：绝不使用 super().__init__(**kwargs)
        default_keys = default_keys0 if not kwargs else kwargs.get('keys')
        self.ybf001_keys = default_keys

    @property
    def name(self):
        return "ybf001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ybf001_keys:
            # 只做参数解包，绝对没有因子业务逻辑
            # dk = (window, ewm) 来自 default_keys0
            ybf001 = calc_ybf001(open=kl_pd['open'],
                                 high=kl_pd['high'],
                                 low=kl_pd['low'],
                                 close=kl_pd['close'],
                                 volume=kl_pd['volume'],
                                 openint=kl_pd['openint'],
                                 window=dk[0],
                                 ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            ybf001 = self._format(ybf001, name=name)
            impulse_dict[name] = ybf001
        return impulse_dict