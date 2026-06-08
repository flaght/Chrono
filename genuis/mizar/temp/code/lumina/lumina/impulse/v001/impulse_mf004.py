from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf004 import mf004 as calc_mf004
import pdb


class ImpulseMf004(ImpulseBase):
    """
    衡量超⼤单卖出⾏为对股价的{}期负⾯压⼒
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf004_keys = default_keys

    @property
    def name(self):
        return "mf004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf004_keys:
            mf004 = calc_mf004(outflowXLRate=kl_pd['outflowXLRate'],
                               ret=kl_pd['ret'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf004 = self._format(mf004,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = mf004
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf004_keys[0][1])
