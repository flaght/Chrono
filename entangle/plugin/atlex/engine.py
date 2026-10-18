import pdb
from toolix.macro.contract import MAIN_CONTRACT_MAPPING, SYMBOL_IMPULSE_MAPPING
from .factorx import Factorx


class Engine(object):

    def __init__(self, codes):
        self._factors_sets = {}
        for code in codes:
            if code in MAIN_CONTRACT_MAPPING:
                self._factors_sets[MAIN_CONTRACT_MAPPING[code]] = Factorx(
                    symbol=MAIN_CONTRACT_MAPPING[code], n_job=4, impulse=SYMBOL_IMPULSE_MAPPING[code].split(','))

    def run(self, symbol, trade_time):
        self._factors_sets[symbol].impluse_run(trade_time=trade_time)
