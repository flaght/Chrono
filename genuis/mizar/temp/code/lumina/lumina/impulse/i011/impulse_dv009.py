import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i011.core.dv009 import dv009 as calc_dv009


class ImpulseDv009(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.dv009_keys = default_keys

    @property
    def name(self):
        return "dv009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.dv009_keys:
            dv009 = calc_dv009(value=kl_pd['value'],
                               close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            dv009 = self._format(dv009, name=name)
            impulse_dict[name] = dv009
        return impulse_dict
