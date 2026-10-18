from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mo008 import mo008 as calc_mo008


class ImpulseMo008(ImpulseBase):
    """期货会员持空头仓占总持仓比率的变化率。即(期货会员持多头仓/总持仓) / ({0}期货会员持多头仓/{0}总持仓)  """ 

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mo008_keys = default_keys

    @property
    def name(self):
        return "mo008"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mo008_keys:
            mo008 = calc_mo008(short=kl_pd['short'],
                               openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            mo008 = self._format(mo008,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = mo008
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mo008_keys[0][1])
