import os
from .signalor import Signalor


class Engine(object):

    def __init__(self, codes):
        self._actuator_sets = {}
        for code in codes:
            if code in os.environ['MAIN_CONTRACT_MAPPING']:
                self._actuator_sets[os.environ['MAIN_CONTRACT_MAPPING']
                                    [code]] = Signalor(code=code)

    def run(self, symbol, trade_time):
        self._actuator_sets[symbol].run(trade_time=trade_time)
