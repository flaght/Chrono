from lumina.impulse.base import ImpulseBase, default_key0
from lumina.impulse.v001.core.cp002 import cp002 as calc_cp002


class ImpulseCp002(ImpulseBase):
    """
    最高筹码峰所占的筹码百分比，值越高，该价位的支撑/阻力效应越强
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key0)

    @property
    def name(self):
        return "cp002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp002(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'])
            name = f"{self.name}"
            alpha = self._format(alpha, name=name, desc=self.__class__.__doc__)
            impulse_dict[name] = alpha
        return impulse_dict
