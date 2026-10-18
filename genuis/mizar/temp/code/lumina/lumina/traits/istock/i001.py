# -*- encoding:utf-8 -*-
import pdb
import pandas as pd
from lumina.traits.base import TraitsBase
import lumina.impulse.i001 as i00


def features():
    f1 = [{
        'ixy001': [(10, 15, 1), (10, 15, 0)]
    }, {
        'ixy002': [(10, 15, 0)]
    }, {
        'ixy003': [(10, 15, 0), (10, 15, 1)]
    }]
    return f1


class Traits001(TraitsBase):

    def __init__(self, impulse=None, all=False):
        if all:
            impulse = i00.__all__
        else:
            impulse = features() if not isinstance(impulse, list) else impulse
        super(Traits001, self).__init__(impulse, i00)

    def create(self, data):
        result = super(Traits001, self).create(data)
        return result

    def tv006(self, data):
        return data.fillna(0)

    def run(self, data):
        data1 = self.create(data)
        results = self.transform(data1)
        return results
