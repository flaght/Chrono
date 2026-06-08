from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.ht004 import ht004 as calc_ht004
import pdb


class ImpulseHt004(ImpulseBase):
    """
    MACD: {}期{}数据快指数移动平均线 {}期{}数据满指数移动平均线 {}期{}数据信号均线
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.ht004_keys = default_keys

    @property
    def name(self):
        return "ht004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ht004_keys:
            macd, signal, hist = calc_ht004(close=kl_pd[dk[0]],
                               window=dk[1],
                               fast=dk[2],
                               slow=dk[3],
                               weriod=dk[4],
                               ewm=True if dk[5] == 1 else False)
            name_macd = "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}".format(self.name, dk[0], dk[1], dk[2],
                                                 dk[3], dk[4], dk[5], 'macd')
            name_signal = "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}".format(self.name, dk[0], dk[1], dk[2],
                                                   dk[3], dk[4], dk[5],
                                                   'signal')
            name_hist = "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}".format(self.name, dk[0], dk[1], dk[2],
                                                 dk[3], dk[4], dk[5], 'hist')
            macd = self._format(macd,
                                name=name_macd,
                                desc=self.__class__.__doc__.format(
                                    dk[2], dk[0], dk[3], dk[0], dk[4], dk[0]))
            signal = self._format(signal,
                                  name=name_signal,
                                  desc=self.__class__.__doc__.format(
                                      dk[2], dk[0], dk[3], dk[0], dk[4], dk[0]))
            hist = self._format(hist,
                                 name=name_hist,
                                 desc=self.__class__.__doc__.format(
                                     dk[2], dk[0], dk[3], dk[0], dk[4], dk[0]))
            impulse_dict[name_macd] = macd
            impulse_dict[name_signal] = signal
            impulse_dict[name_hist] = hist
        return impulse_dict

    # def description(self, base_str=None):
    #     base_str = self.__class__.__doc__ if base_str is None else base_str
    #     return base_str.format(self.ht004_keys[0][1], self.ht004_keys[0][0])
