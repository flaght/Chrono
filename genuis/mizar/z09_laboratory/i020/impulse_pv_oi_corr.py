# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.pv_oi_corr import pv_oi_corr as calc_pv_oi_corr


class ImpulsePvOiCorr(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.pv_oi_corr_keys = default_keys

    @property
    def name(self):
        return "pv_oi_corr"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.pv_oi_corr_keys:
            # 仅做参数解包，无任何因子业务逻辑
            alpha = calc_pv_oi_corr(close=kl_pd['close'],
                                    openint=kl_pd['openint'],
                                    volume=kl_pd['volume'],
                                    window=dk[0],
                                    weriod=dk[1],
                                    ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            alpha = self._format(alpha, name=name)
            impulse_dict[name] = alpha
        return impulse_dict