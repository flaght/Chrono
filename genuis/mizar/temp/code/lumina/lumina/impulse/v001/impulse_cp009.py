from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp009 import cp009 as calc_cp009


class ImpulseCp009(ImpulseBase):
    """
    已套牢投资者的{}期平均亏损率，衡量亏损方的“痛苦指数”。
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp009(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict
