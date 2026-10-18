import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i009.core.iv006 import iv006 as calc_tv006


class ImpulseIv006(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.iv006_keys = default_keys

    @property
    def name(self):
        return "iv006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.iv006_keys:
            iv006 = calc_tv006(value=kl_pd['value'],
                               close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            iv006 = self._format(iv006, name=name)
            impulse_dict[name] = iv006
        return impulse_dict
