from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mo009 import mo009 as calc_mo009


class ImpulseMo009(ImpulseBase):
    """ (期货会员持多头仓和持空头仓的差值)占总持仓比率  """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else [(1, 0, 1)
                                                        ]  #kwargs.get('keys')
        self.mo009_keys = default_keys

    @property
    def name(self):
        return "mo009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mo009_keys:
            mo009 = calc_mo009(long=kl_pd['long'],
                               short=kl_pd['short'],
                               openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            mo009 = self._format(mo009,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = mo009
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__  if base_str is None else base_str
        return base_str.format(self.mo009_keys[0][1])
