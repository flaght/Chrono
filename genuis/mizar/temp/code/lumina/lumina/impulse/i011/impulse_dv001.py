import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i011.core.dv001 import dv001 as calc_dv001

class ImpulseDv001(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.dv001_keys = default_keys

    @property
    def name(self):
        return "dv001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.dv001_keys:
            dv001 = calc_dv001(value=kl_pd['value'],
                              window=dk[0],
                              weriod=dk[1],
                              ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            dv001 = self._format(dv001, name=name)
            impulse_dict[name] = dv001
        return impulse_dict