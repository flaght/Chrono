from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.hm005 import hm005 as calc_hm005
import pdb


class ImpulseHm005(ImpulseBase):
    """
    衡量龙虎榜过去{}个交易日的累积净买入额，反映短期资金的真实流向趋势
    """
    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.ht001_keys = default_keys

    @property
    def name(self):
        return "hm005"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ht001_keys:
            ht001 = calc_hm005(buy=kl_pd['buy'],
                               sell=kl_pd['sell'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ht001 = self._format(ht001,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = ht001
        return impulse_dict