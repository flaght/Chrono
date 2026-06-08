from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.hm010 import hm010 as calc_hm010
import pdb


class ImpulseHm010(ImpulseBase):
    """
    计算股票连续登上龙虎榜的天数，并进行{}日平滑。
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.hm010_keys = default_keys

    @property
    def name(self):
        return "hm010"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.hm010_keys:
            # 注意：输入数据是 kl_pd['on_list']
            hm010_val = calc_hm010(on_list=kl_pd['on_list'],
                                   window=dk[0],
                                   weriod=dk[1], # weriod 在此因子中未直接使用，但为保持接口一致性而传入
                                   ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            hm010_val = self._format(hm010_val,
                                     name=name,
                                     desc=self.__class__.__doc__.format(dk[0])) # 使用window(dk[0])进行描述
            impulse_dict[name] = hm010_val
        return impulse_dict