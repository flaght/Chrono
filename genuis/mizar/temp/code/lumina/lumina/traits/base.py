# -*- encoding:utf-8 -*-
import pdb


class TraitsBase(object):

    def __init__(self, impulse, cls):
        self._impulse = impulse
        self._class = cls

    def transform(self, results):
        res = {}
        for k, v in results.items():
            name = k.split('_')[0]
            if hasattr(self, name):
                func = getattr(self, name)
                v = func(v)
            res[k] = v
        return res

    def create(self, data):
        res = {}
        for f in self._impulse:
            if isinstance(f, dict):
                name = list(f.keys())[0]
                name = "Impulse{0}".format(name.capitalize())
                cls = getattr(self._class, name)
                obj = cls(keys=list(f.values())[0])
            else:
                cls = getattr(self._class, f)
                obj = cls()
            result = obj.calc_impulse(data.copy())
            for k, v in result.items():
                res[k] = result[k]
        return res
