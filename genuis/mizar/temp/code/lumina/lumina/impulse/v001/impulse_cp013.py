from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp013 import cp013 as calc_cp013

class ImpulseCp013(ImpulseBase):
    """
    即获利盘比例，也代表了{}期价格下方支撑盘的厚度，是股价下跌的缓冲垫
    """
    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp013"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp013(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha, name=name, desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict