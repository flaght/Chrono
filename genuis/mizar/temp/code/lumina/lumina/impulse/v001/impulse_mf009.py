from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf009 import mf009 as calc_mf009
import pdb


class ImpulseMf009(ImpulseBase):
    """
    衡量散户资⾦⾏为与股价⾛势的{} 期同步性。 
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf009_keys = default_keys  

    @property
    def name(self):
        return "mf009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf009_keys:
            mf009 = calc_mf009(smainFlow=kl_pd['smainFlow'],
                               ret=kl_pd['ret'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf009 = self._format(mf009,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = mf009
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf009_keys[0][0])
