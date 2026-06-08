from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key5
from lumina.impulse.v001.core.in001 import in001 as calc_in001
import pdb


class ImpulseIn001(ImpulseBase):
    """
    基于前一日计算的关键支撑/阻力水平
    """

    def __init__(self, **kwargs):
        default_keys = default_key5 if not kwargs else kwargs.get('keys')
        self.in001_keys = default_keys

    @property
    def name(self):
        return "in001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.in001_keys:
            pp, r1, s1, r2, s2, r3, s3 = calc_in001(
                high=kl_pd['high'],
                low=kl_pd['low'],
                close=kl_pd['close'],
                window=dk[0],
                ewm=True if dk[1] == 1 else False)
            name_pp = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], 'pp')
            name_r1 = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], 'r1')
            name_s1 = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], 's1')
            name_r2 = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], 'r2')
            name_s2 = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], 's2')
            name_r3 = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], 'r3')
            name_s3 = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], 's3')
            pp = self._format(pp, name=name_pp, desc=self.__class__.__doc__)
            r1 = self._format(r1, name=name_r1, desc=self.__class__.__doc__)
            s1 = self._format(s1, name=name_s1, desc=self.__class__.__doc__)
            r2 = self._format(r2, name=name_r2, desc=self.__class__.__doc__)
            s2 = self._format(s2, name=name_s2, desc=self.__class__.__doc__)
            r3 = self._format(r3, name=name_r3, desc=self.__class__.__doc__)
            s3 = self._format(s3, name=name_s3, desc=self.__class__.__doc__)
            impulse_dict[name_pp] = pp
            impulse_dict[name_r1] = r1
            impulse_dict[name_s1] = s1
            impulse_dict[name_r2] = r2
            impulse_dict[name_s2] = s2
            impulse_dict[name_r3] = r3
            impulse_dict[name_s3] = s3
        return impulse_dict
