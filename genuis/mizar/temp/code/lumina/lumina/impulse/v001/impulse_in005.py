from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key6
from lumina.impulse.v001.core.in005 import in005 as calc_in005
import pdb


class ImpulseIn005(ImpulseBase):
    '''
    MACD: {}期快指数移动平均线 {}期满指数移动平均线 {}期信号均线
    '''

    def __init__(self, **kwargs):
        default_keys = default_key6 if not kwargs else kwargs.get('keys')
        self.in005_keys = default_keys

    @property
    def name(self):
        return "in005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.in005_keys:
            macd, signal, hist = calc_in005(close=kl_pd['close'],
                                            window=dk[0],
                                            fast=dk[1],
                                            slow=dk[2],
                                            weriod=dk[3],
                                            ewm=True if dk[4] == 1 else False)
            name_macd = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1],
                                                 dk[2], dk[3], dk[4], 'macd')
            name_signal = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1],
                                                   dk[2], dk[3], dk[4],
                                                   'signal')
            name_hist = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1],
                                                 dk[2], dk[3], dk[4], 'hist')
            macd = self._format(macd,
                                name=name_macd,
                                desc=self.__class__.__doc__.format(
                                    dk[1], dk[2], dk[3]))
            signal = self._format(signal,
                                  name=name_signal,
                                  desc=self.__class__.__doc__.format(
                                      dk[1], dk[2], dk[3]))
            hist = self._format(hist,
                                name=name_hist,
                                desc=self.__class__.__doc__.format(
                                    dk[1], dk[2], dk[3]))
            impulse_dict[name_macd] = macd
            impulse_dict[name_signal] = signal
            impulse_dict[name_hist] = hist
        return impulse_dict
