import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi038 import oi038 as calc_oi038


class ImpulseOi038(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi038_keys = default_keys

    @property
    def name(self):
        return "oi038"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi038_keys:
            oi038 = calc_oi038(close=kl_pd['close'],
                               openint=kl_pd['openint'],
                               value=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi038 = self._format(oi038, name=name)
            impulse_dict[name] = oi038
        return impulse_dict
