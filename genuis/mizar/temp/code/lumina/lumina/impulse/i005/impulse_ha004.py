import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i005.core.ha004 import ha004 as calc_ha004


class ImpulseHa004(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ha004_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "ha004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ha004_keys:
            ha004 = calc_ha004(value=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ha004 = self._format(ha004, name=name)
            impulse_dict[name] = ha004
        return impulse_dict