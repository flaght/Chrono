from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp008 import cp008 as calc_cp008


class ImpulseCp008(ImpulseBase):
    """
    {}期平均价格相对于全市场加权平均持仓成本（ASR）的偏离度，反映市场整体盈亏状态。
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp008"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp008(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = alpha
        return impulse_dict
