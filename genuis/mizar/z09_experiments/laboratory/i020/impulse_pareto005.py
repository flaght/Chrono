import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from laboratory.i020.core.pareto005 import pareto005 as calc_pareto005

class ImpulsePareto005(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.pareto005_keys = default_keys

    @property
    def name(self):
        return "pareto005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.pareto005_keys:
            # dk = (window, fast, slow, ewm)
            pareto005 = calc_pareto005(openint=kl_pd['openint'],
                                       close=kl_pd['close'],
                                       volume=kl_pd['volume'],
                                       window=dk[0],
                                       fast=dk[1],
                                       slow=dk[2],
                                       ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2], dk[3])
            pareto005 = self._format(pareto005, name=name)
            impulse_dict[name] = pareto005
        return impulse_dict