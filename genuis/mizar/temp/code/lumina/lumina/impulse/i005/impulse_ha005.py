import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i005.core.ha005 import ha005 as calc_ha005


class ImpulseHa005(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ha005_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "ha005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ha005_keys:
            ha005 = calc_ha005(value=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ha005 = self._format(ha005, name=name)
            impulse_dict[name] = ha005
        return impulse_dict
