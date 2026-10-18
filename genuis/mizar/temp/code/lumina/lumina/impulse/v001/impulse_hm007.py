from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.hm007 import hm007 as calc_hm007
import pdb


class ImpulseHm007(ImpulseBase):
    """
    ## 龙虎榜当日成交量与其过去N日平均成交量的比值，衡量交易活跃度的异常放大程度
    """
    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.ht001_keys = default_keys

    @property
    def name(self):
        return "hm007"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ht001_keys:
            ht001 = calc_hm007(hotvalue=kl_pd['hotvalue'],
                               value=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ht001 = self._format(ht001,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = ht001
        return impulse_dict