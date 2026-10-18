from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf003 import mf003 as calc_mf003
import pdb


class ImpulseMf003(ImpulseBase):
    """
    衡量主⼒资⾦净流⼊{} 期的持续性。 
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf003_keys = default_keys

    @property
    def name(self):
        return "mf003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf003_keys:
            mf003 = calc_mf003(mainFlowRate=kl_pd['mainFlowRate'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf003 = self._format(mf003,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = mf003
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf003_keys[0][1])
