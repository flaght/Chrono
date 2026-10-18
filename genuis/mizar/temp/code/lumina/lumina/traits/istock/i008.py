# -*- encoding:utf-8 -*-
import pdb
import pandas as pd
from lumina.traits.base import TraitsBase
import lumina.impulse.i008 as i00


class Traits008(TraitsBase):

    def __init__(self, impulse=None, all=False):
        if all:
            impulse = i00.__all__
        else:
            impulse = ['ImpulseTv001', 'ImpulseTv002', 'ImpulseTv006'
                       ] if not isinstance(impulse, list) else impulse
        super(Traits008, self).__init__(impulse, i00)

    def create(self, data):
        result = super(Traits008, self).create(data)
        return result

    def tv006(self, data):
        return data.fillna(0)

    def run(self, data):
        data1 = self.create(data)
        results = self.transform(data1)
        return results
