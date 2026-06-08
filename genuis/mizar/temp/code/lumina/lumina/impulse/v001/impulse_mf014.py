from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf014 import mf014 as calc_mf014
import pdb


class ImpulseMf014(ImpulseBase):
    """
    过去{}日⼤单隐匿指数。
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf014_keys = default_keys  

    @property
    def name(self):
        return "mf014"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf014_keys:
            mf014 = calc_mf014(inflowL=kl_pd['inflowL'],
                               inflowXL=kl_pd['inflowXL'],
                               buyOrdL=kl_pd['buyOrdL'],
                               buyOrdXL=kl_pd['buyOrdXL'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf014 = self._format(mf014,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = mf014
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf014_keys[0][0])
