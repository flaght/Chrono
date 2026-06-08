from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.hm009 import hm009 as calc_hm009
import pdb


class ImpulseHm009(ImpulseBase):
    """
    过去{}个交易日中登上龙虎榜的总天数，衡量个股在近期被游资关注的频繁程度。
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.hm009_keys = default_keys

    @property
    def name(self):
        return "hm009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.hm009_keys:
            # 注意：输入数据是 kl_pd['on_list']
            hm009_val = calc_hm009(on_list=kl_pd['on_list'],
                                   window=dk[0],
                                   weriod=dk[1],
                                   ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            hm009_val = self._format(hm009_val,
                                     name=name,
                                     desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = hm009_val
        return impulse_dict