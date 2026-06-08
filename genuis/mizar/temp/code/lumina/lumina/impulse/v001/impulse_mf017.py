from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mf017 import mf017 as calc_mf017
import pdb


class ImpulseMf017(ImpulseBase):
    """
    滚动{}期散户投降信号和。
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mf017_keys = default_keys  

    @property
    def name(self):
        return "mf017"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mf017_keys:
            mf017 = calc_mf017(smainFlowRate=kl_pd['smainFlowRate'],
                               ret=kl_pd['ret'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            mf017 = self._format(mf017,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = mf017
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mf017_keys[0][0])
