from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf016 import mf016 as calc_mf016
import pdb


class ImpulseMf016(ImpulseBase):
    """
    量价资⾦流背离{}期平均值。
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf016_keys = default_keys  

    @property
    def name(self):
        return "mf016"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf016_keys:
            mf016 = calc_mf016(mainFlowRate=kl_pd['mainFlowRate'],
                               mainBuyVol=kl_pd['mainBuyVol'],
                               mainSellVol=kl_pd['mainSellVol'],
                               turnoverVol=kl_pd['turnoverVol'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf016 = self._format(mf016,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = mf016
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf016_keys[0][0])
