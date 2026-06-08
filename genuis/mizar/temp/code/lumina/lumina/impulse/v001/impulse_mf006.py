from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf006 import mf006 as calc_mf006
import pdb


class ImpulseMf006(ImpulseBase):
    """
    衡量超⼤单的{}期买卖⼒量对⽐。 
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf006_keys = default_keys

    @property
    def name(self):
        return "mf006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf006_keys:
            mf006 = calc_mf006(inflowXL=kl_pd['inflowXL'],
                               outflowXL=kl_pd['outflowXL'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf006 = self._format(mf006,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = mf006
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf006_keys[0][0])
