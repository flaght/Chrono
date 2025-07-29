import time, pdb, os
import pandas as pd
from collections import OrderedDict
from alphacopilot.api.calendars import *
from cn_futures import CNFutures

from dolphin import Dolphin


class Fantastic(object):

    def __init__(self, code, start_date, end_date, uri, session=0):
        self._session = int(time.time()) if session == 0 else session
        self.history_file = OrderedDict()
        self._start_date = start_date
        self._end_date = end_date
        self._code = code
        self.init_strategies()
        self.engine = CNFutures(code=code,
                                uri=uri,
                                strategies_pool=self.strategies_pool)

    def init_strategies(self):  ## 初始化策略池
        self.strategies_pool = OrderedDict()
        self.strategies_pool['10001'] = Dolphin(strategy_id='10001', params={})

    def start(self):
        dates = makeSchedule(self._start_date,
                             endDate=self._end_date,
                             calendar='China.SSE',
                             tenor='1b')
        for d in dates:
            self.engine.start(trade_date=d.strftime('%Y-%m-%d'))


uri = os.path.join("temp")
s1 = Fantastic(code='IM',
               start_date='2025-03-03',
               end_date='2025-04-30',
               uri=uri)
s1.start()
