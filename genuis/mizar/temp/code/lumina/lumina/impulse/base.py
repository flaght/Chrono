from lumina.env import *
from lumina.impulse.env import *
import pdb, os
import pandas as pd
import pdb

class ImpulseBase(object):

    def __init__(self, **kwargs):
        # 子类继续完成自有的构造
        self._init_self(**kwargs)

    def _init_self(self, **kwargs):
        """子类因子针对可扩展参数的初始化"""
        pass

    def _format(self, data, name, desc=None):
        if g_format == 2:
            return self._format2(data, name, desc)
        elif g_format == 3:
            return self._format3(data, name)

    def _format2(self, data, name, desc):
        data = data.stack()
        data.name = name
        if isinstance(desc, str):
            data.desc = desc
        return data

    def _format3(self, data, name):
        data = data.stack()
        data.name = 'value'

        ## 重建索引
        new_index = data.index.to_frame(index=False)
        new_index['name'] = name
        new_index = pd.MultiIndex.from_frame(new_index)
        data.index = new_index

        return data

    def default_dependencies(self):
        return {
            'close': 'minClosePrice',
            'low': 'minLowPrice',
            'high': 'minHighPrice',
            'open': 'minOpenPrice',
            'volume': 'minTurnoverVol',
            'value': 'minTurnoverValue',
            'chg': 'minLogRet'
        }

    def external_dependencies(self):
        return {
            'close1': 'minClosePrice',
            'low1': 'minLowPrice',
            'high2': 'minHighPrice',
            'open': 'minOpenPrice',
            'volume': 'minTurnoverVol',
            'value': 'minTurnoverValue',
            'chg': 'minLogRet'
        }

    @classmethod
    def serializ(cls, params):
        return params
