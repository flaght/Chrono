import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i009.core.iv003 import iv003 as calc_tv003


class ImpulseIv003(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.iv003_keys = default_keys

    @property
    def name(self):
        return "iv003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.iv003_keys:
            iv003 = calc_tv003(close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            iv003 = self._format(iv003, name=name)
            impulse_dict[name] = iv003
        return impulse_dict
