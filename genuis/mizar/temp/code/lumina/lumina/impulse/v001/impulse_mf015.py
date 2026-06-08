from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf015 import mf015 as calc_mf015
import pdb


class ImpulseMf015(ImpulseBase):
    """
    第{}天资⾦换⼿（对敲）指数。 
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf015_keys = default_keys  

    @property
    def name(self):
        return "mf015"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf015_keys:
            mf015 = calc_mf015(inflow=kl_pd['inflow'],
                               outflow=kl_pd['outflow'],
                               netFlow=kl_pd['netFlow'],
                               window=dk[0])
            name = "{0}_{1}".format(self.name, dk[0])
            mf015 = self._format(mf015,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]+1))
            impulse_dict[name] = mf015
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf015_keys[0][0])
