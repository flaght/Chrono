import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi010 import oi010 as calc_oi010


class ImpulseOi010(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi010_keys = default_keys

    @property
    def name(self):
        return "oi010"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi010_keys:
            oi010 = calc_oi010(openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi010 = self._format(oi010, name=name)
            impulse_dict[name] = oi010
        return impulse_dict
