from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf008 import mf008 as calc_mf008
import pdb


class ImpulseMf008(ImpulseBase):
    """
    衡量主⼒资⾦⾏为与股价⾛势的{} 期同步性。 
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf008_keys = default_keys

    @property
    def name(self):
        return "mf008"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf008_keys:
            mf008 = calc_mf008(mainFlow=kl_pd['mainFlow'],
                               ret=kl_pd['ret'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf008 = self._format(mf008,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = mf008
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf008_keys[0][0])
