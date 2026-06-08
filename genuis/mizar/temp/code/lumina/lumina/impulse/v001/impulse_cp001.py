import pdb
from lumina.impulse.base import ImpulseBase, default_key0
from lumina.impulse.v001.core.cp001 import cp001 as calc_cp001


class ImpulseCp001(ImpulseBase):
    """
    筹码最密集的成本价，代表市场核心成本区，
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key0)

    @property
    def name(self):
        return "cp001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp001(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'])
            name = f"{self.name}"
            alpha = self._format(alpha, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = alpha
        return impulse_dict
