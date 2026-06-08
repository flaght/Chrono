from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf002 import mf002 as calc_mf002
import pdb


class ImpulseMf002(ImpulseBase):
    """
    衡量⼩单净流⼊占成交额⽐例的{}期变化趋势。 
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf002_keys = default_keys

    @property
    def name(self):
        return "mf002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf002_keys:
            mf002 = calc_mf002(netFlowSRate=kl_pd['netFlowSRate'],
                               fast_window=dk[0],
                               slow_window=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            mf002 = self._format(mf002,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = mf002
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf002_keys[0][1])
