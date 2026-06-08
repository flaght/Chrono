from lumina.impulse.base import ImpulseBase, default_key0
from lumina.impulse.v001.core.cp020 import cp020 as calc_cp020


class ImpulseCp020(ImpulseBase):
    """
    全市场平均成本(ASR)与最集中成本(Peak)的偏离
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key0)

    @property
    def name(self):
        return "cp020"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp020(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'])
            name = f"{self.name}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__)
            impulse_dict[name] = alpha
        return impulse_dict
