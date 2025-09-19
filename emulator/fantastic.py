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
        
        # --- 新增：定义手续费率 ---
        # 费率通常是按万分之几计算的。例如，万分之0.23 (0.23 bps) 就是 0.000023。
        # 假设IM合约的开仓和平仓手续费率都是成交额的万分之0.23
        im_commission_rate = {
            'open': 0.000023, 
            'close': 0.000023
        }

        # 定义策略参数，包括初始资金、合约乘数和手续费率
        strategy_params = {
            'initial_capital': 1_000_000,
            'multiplier': 200,  # IM合约乘数是200
            'commission_rate': im_commission_rate # 将费率字典传入
        }
        self.strategies_pool['10001'] = Dolphin(strategy_id='10001', params=strategy_params)

    def start(self):
        dates = makeSchedule(self._start_date,
                             endDate=self._end_date,
                             calendar='China.SSE',
                             tenor='1b')
        for d in dates:
            print(f"Backtesting for date: {d.strftime('%Y-%m-%d')}...")
            self.engine.start(trade_date=d.strftime('%Y-%m-%d'))

        # 回测循环结束后，调用结果分析函数
        print("\nBacktest finished. Calculating final performance...")
        for name, strategy in self.engine.strategies_pool.items():
            strategy.calc_result()


# 确保你的数据文件路径正确
uri = "temp"
s1 = Fantastic(code='IM',
               start_date='2025-03-05',
               end_date='2025-04-30',
               uri=uri)
s1.start()