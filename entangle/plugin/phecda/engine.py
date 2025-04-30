import pdb
from toolix.macro.contract import MAIN_CONTRACT_MAPPING, CHAOS_PHECDA_MAPPING
from .signalor import Signalor


class Engine(object):

    def __init__(self, codes):
        self._actuator_sets = {}
        for code in codes:
            if code in MAIN_CONTRACT_MAPPING:
                ### 使用list symbol--> 一个合约对应多个信号策略 ### 待修改
                self._actuator_sets[MAIN_CONTRACT_MAPPING[code]] = Signalor(
                    id=CHAOS_PHECDA_MAPPING[MAIN_CONTRACT_MAPPING[code]],
                    code=code)

    def run(self, symbol, trade_time):
        if symbol in self._actuator_sets:
            self._actuator_sets[symbol].run(trade_time=trade_time)
