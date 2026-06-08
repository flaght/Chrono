from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf005 import mf005 as calc_mf005
import pdb


class ImpulseMf005(ImpulseBase):
    """
    衡量主⼒资⾦净流⼊的{} 期加速度。 
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf005_keys = default_keys

    @property
    def name(self):
        return "mf005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf005_keys:
            mf005 = calc_mf005(mainFlowRate=kl_pd['mainFlowRate'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf005 = self._format(mf005,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = mf005
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf005_keys[0][0])
