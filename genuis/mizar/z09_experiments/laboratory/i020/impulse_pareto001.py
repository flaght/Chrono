# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from laboratory.i020.core.pareto001 import pareto001 as calc_pareto001

class ImpulsePareto001(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.pareto001_keys = default_keys

    @property
    def name(self):
        return "pareto001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.pareto001_keys:
            # dk = (window, fast, slow, ewm)
            pareto001 = calc_pareto001(volume=kl_pd['volume'],
                                        window=dk[0],
                                        fast=dk[1],
                                        slow=dk[2],
                                        ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2], dk[3])
            pareto001 = self._format(pareto001, name=name)
            impulse_dict[name] = pareto001
        return impulse_dict