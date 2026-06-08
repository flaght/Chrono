from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf010 import mf010 as calc_mf010
import pdb


class ImpulseMf010(ImpulseBase):
    """
    衡量主⼒资⾦流⼊中⽆法被当⽇股价涨跌解释的{}日滚动“超预期”部分。
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf010_keys = default_keys  

    @property
    def name(self):
        return "mf010"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf010_keys:
            mf010 = calc_mf010(mainFlow=kl_pd['mainFlow'],
                               ret=kl_pd['ret'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf010 = self._format(mf010,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = mf010
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf010_keys[0][0])
