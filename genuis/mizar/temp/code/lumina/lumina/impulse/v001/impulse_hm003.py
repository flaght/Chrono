from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.hm003 import hm003 as calc_hm003
import pdb


class ImpulseHm003(ImpulseBase):
    """
    衡量龙虎榜上{0}日买方总金额与{0}卖方总金额的比值，直观反映多空双方的资金实力对比。
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.ht003_keys = default_keys

    @property
    def name(self):
        return "hm003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ht003_keys:
            ht001 = calc_hm003(buy=kl_pd['buy'],
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
