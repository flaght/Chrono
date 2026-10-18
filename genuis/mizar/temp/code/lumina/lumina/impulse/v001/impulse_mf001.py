from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf001 import mf001 as calc_mf001
import pdb


class ImpulseMf001(ImpulseBase):
    """
    衡量主⼒资⾦与散户资⾦流向{} 期的背离动量。 
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf001_keys = default_keys

    @property
    def name(self):
        return "mf001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf001_keys:
            mf001 = calc_mf001(mainFlow=kl_pd['mainFlow'],
                               smainFlow=kl_pd['smainFlow'],
                               fast_window=dk[0],
                               slow_window=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            mf001 = self._format(mf001,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = mf001
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf001_keys[0][1])
