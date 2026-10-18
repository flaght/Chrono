import pdb
from toolix.macro.contract import OPTIONS_SPOT, MAIN_CONTRACT_MAPPING
from .signalor import Signalor


class Engine(object):

    def __init__(self, symbols):
        self._actuator_sets = {}
        for symbol in symbols:
            parts = symbol.split('-')
            underlying_code_prefix = parts[0][:2]
            if underlying_code_prefix in OPTIONS_SPOT:
                self._actuator_sets[symbol] = Signalor(
                    symbol=symbol,
                    spot_symbol=MAIN_CONTRACT_MAPPING[
                        OPTIONS_SPOT[underlying_code_prefix]])

    def run(self, symbol, trade_time):
        print()
        
    def on_tick(self, data):
        symbol = data['symbol']
        if symbol in self._actuator_sets:
            self._actuator_sets[symbol].on_tick(data=data)
