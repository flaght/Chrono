from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.hm006 import hm006 as calc_hm006
import pdb


class ImpulseHm006(ImpulseBase):
    """
    龙虎榜净流入{}日变化率 正值表示买入力度在加速增强，可能是行情启动信号；负值则表示动能衰减，需要警惕
    """
    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.ht001_keys = default_keys

    @property
    def name(self):
        return "hm002"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ht001_keys:
            ht001 = calc_hm006(buy=kl_pd['buy'],
                               sell=kl_pd['sell'],
                               hotvalue=kl_pd['hotvalue'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ht001 = self._format(ht001,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = ht001
        return impulse_dict