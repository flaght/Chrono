import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i005.core.ha003 import ha003 as calc_ha003


class ImpulseHa003(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ha003_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "ha003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ha003_keys:
            ha003 = calc_ha003(value=kl_pd['value'] / 1e6,
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ha003 = self._format(ha003, name=name)
            impulse_dict[name] = ha003
        return impulse_dict
