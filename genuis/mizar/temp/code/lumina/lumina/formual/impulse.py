import re, pdb
import pandas as pd
from collections import defaultdict
import lumina.impulse as impulse


class Impulse(object):

    def __init__(self, formuals):
        self.pattern = re.compile(
            r'([a-zA-Z]+\d+)_(\d+)_(\d+)(?:_(\d+))?(?:_(\d+))?(?:_(\d+))?')
        self.formuals = formuals
        self.init_obj()

    def init_obj(self):
        result = defaultdict(list)
        for item in self.formuals:
            match = self.pattern.match(item)
            if match:
                prefix = match.group(1)
                numbers = tuple(
                    int(num) for num in match.groups()[1:] if num is not None)
                result[prefix].append(numbers)
            else:
                print('No match:', item)
        self.factors = [{key: value} for key, value in result.items()]
        self._max_window = max(max(max(t) for t in list(d.values())[0]) for d in self.factors)

    def max_window(self):
        return int(self._max_window * 1.2)

    def batch(self, data):
        res = []
        for f in self.factors:
            if isinstance(f, dict):
                name = list(f.keys())[0]
                name = "Impulse{0}".format(name.capitalize())
                cls = getattr(impulse, name)
                params = cls.serializ(list(f.values())[0])
                obj = cls(keys=params)
            else:
                cls = getattr(impulse, f)
                obj = cls()
            res += obj.calc_impulse(data).values()
        
        data = pd.concat(res,axis=1)
        data = data.sort_index()
        return data
