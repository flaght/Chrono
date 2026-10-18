# -*- encoding:utf-8 -*-
import pdb
from lumina.position.base import PositionBase

class FakePosition(PositionBase):
    """Faker仓位管理类"""
    def fit_weight(self, factor_object):
        return 1
    
    def fit_position(self, factor_object):
        return -1
        
    def _init_self(self, **kwargs):
        pass