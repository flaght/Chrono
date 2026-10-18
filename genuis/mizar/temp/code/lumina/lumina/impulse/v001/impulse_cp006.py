from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp006 import cp006 as calc_cp006


class ImpulseCp006(ImpulseBase):
    """
    {}期收盘平均价下方所有筹码的总占比。衡量市场浮动盈利情况的核心指标，比例过高可能有回调压力。
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp006(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict
