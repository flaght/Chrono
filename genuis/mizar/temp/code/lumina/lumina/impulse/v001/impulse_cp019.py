from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp019 import cp019 as calc_cp019
import pdb


class ImpulseCp019(ImpulseBase):
    """
    {}期平均价格到下一个显著阻力峰的价格空间百分比。值越大，说明股价上方的“真空地带”越大
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp019"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp019(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict
