import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i008.core.tv007 import tv007 as calc_tv007

class ImpulseTv007(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htv007_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv007"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv007_keys:
            htv007 = calc_tv007(close=kl_pd['close'],
                                volume=kl_pd['volume'],
                                weriod=dk[0],
                                window=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htv007 = self._format(htv007, name=name)
            impulse_dict[name] = htv007
        return impulse_dict