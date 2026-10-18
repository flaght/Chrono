from lumina.impulse.base import ImpulseBase
from lumina.impulse.v001.core.cp004 import cp004 as calc_cp004


class ImpulseCp004(ImpulseBase):
    """
    覆盖中间{}%筹码的价格区间宽度。带宽越窄，市场成本越一致，趋势一旦形成可能越猛烈
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', [(0.8)])

    @property
    def name(self):
        return "cp004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp004(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               percent=dk[0])
            name = f"{self.name}_{dk[0] * 100}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict
