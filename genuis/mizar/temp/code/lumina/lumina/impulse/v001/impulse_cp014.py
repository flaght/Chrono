from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp014 import cp014 as calc_cp014


class ImpulseCp014(ImpulseBase):
    """
     状态因子：+1表示{0}期价格在峰值之上（强势区），-1表示{0}期价格在其之下（弱势区），0在峰值附近（博弈区）
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp014"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp014(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict
