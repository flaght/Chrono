from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf007 import mf007 as calc_mf007
import pdb


class ImpulseMf007(ImpulseBase):
    """
    衡量开盘资⾦与收盘资⾦的{}日流向差异。 
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf007_keys = default_keys

    @property
    def name(self):
        return "mf007"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf007_keys:
            mf007 = calc_mf007(net_in_opn=kl_pd['net_in_opn'],
                               net_in_cls=kl_pd['net_in_cls'],
                               turnoverValue=kl_pd['turnoverValue'],
                               window=dk[0])
            name = "{0}_{1}".format(self.name, dk[0])
            mf007 = self._format(mf007,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]+1))
            impulse_dict[name] = mf007
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf007_keys[0][0])
