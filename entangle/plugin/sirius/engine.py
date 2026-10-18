import pdb
from toolix.macro.contract import MAIN_CONTRACT_MAPPING, CHAOS_SIRIUS_MAPPING
from .signalor import Signalor


class Engine(object):

    def __init__(self, codes):
        self._actuator_sets = {}
        for code in codes:
            if code in MAIN_CONTRACT_MAPPING:
                self._actuator_sets[MAIN_CONTRACT_MAPPING[code]] = Signalor(
                    code=code,
                    symbol=MAIN_CONTRACT_MAPPING[code],
                    task_id=CHAOS_SIRIUS_MAPPING[code])

    def run(self, symbol, trade_time):
        if symbol in self._actuator_sets:
            self._actuator_sets[symbol].run(trade_time=trade_time)
