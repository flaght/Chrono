from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf013 import mf013 as calc_mf013
import pdb


class ImpulseMf013(ImpulseBase):
    """
    主⼒与散户平均每笔订单⾦额⽐{}日平均值。 
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf013_keys = default_keys  

    @property
    def name(self):
        return "mf013"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf013_keys:
            mf013 = calc_mf013(mainInflow=kl_pd['mainInflow'],
                               mainBuyOrd=kl_pd['mainBuyOrd'],
                               inflowS=kl_pd['inflowS'],
                               inflowM=kl_pd['inflowM'],
                               buyOrdS=kl_pd['buyOrdS'],
                               buyOrdM=kl_pd['buyOrdM'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf013 = self._format(mf013,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = mf013
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf013_keys[0][0])
