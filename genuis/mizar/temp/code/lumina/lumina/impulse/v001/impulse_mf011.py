from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf011 import mf011 as calc_mf011
import pdb


class ImpulseMf011(ImpulseBase):
    """
    衡量超⼤单与⼩单资⾦流向的{}期背离程度。 
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf011_keys = default_keys  

    @property
    def name(self):
        return "mf011"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf011_keys:
            mf011 = calc_mf011(netFlowXL=kl_pd['netFlowXL'],
                               netFlowS=kl_pd['netFlowS'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf011 = self._format(mf011,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = mf011
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf011_keys[0][0])
