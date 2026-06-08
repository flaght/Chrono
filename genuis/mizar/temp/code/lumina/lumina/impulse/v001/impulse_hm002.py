from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.hm002 import hm002 as calc_hm002
import pdb


class ImpulseHm002(ImpulseBase):
    """
    衡量龙虎榜席位成交额占{}日总成交额的比例，反映大资金对交易的主导程度。
    """
    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.ht002_keys = default_keys

    @property
    def name(self):
        return "hm002"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ht002_keys:
            ht002 = calc_hm002(buy=kl_pd['buy'],
                               sell=kl_pd['sell'],
                               value=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ht002 = self._format(ht002,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = ht002
        return impulse_dict
