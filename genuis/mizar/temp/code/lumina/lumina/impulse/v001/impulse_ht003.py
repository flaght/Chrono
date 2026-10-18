from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.ht003 import ht003 as calc_ht003
import pdb


class ImpulseHt003(ImpulseBase):
    """
    RSI  {} 期{}数据涨跌幅度的比率评估市场多空力量对比
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.ht003_keys = default_keys

    @property
    def name(self):
        return "ht003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ht003_keys:
            ht003 = calc_ht003(close=kl_pd[dk[0]],
                               window=dk[1],
                               weriod=dk[2],
                               ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[1], dk[2], dk[3])
            ht003 = self._format(ht003,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1], dk[0]))
            impulse_dict[name] = ht003
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.ht003_keys[0][1], self.ht003_keys[0][0])
