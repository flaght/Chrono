import pdb
from toolix.macro.contract import MAIN_CONTRACT_MAPPING, CHAOS_VIRGTOR_MAPPING
from .signalor import Signalor


class Engine(object):

    def __init__(self, codes):
        self._actuator_sets = {}
        for code in codes:
            if code in MAIN_CONTRACT_MAPPING:
                self._actuator_sets[MAIN_CONTRACT_MAPPING[code]] = Signalor(
                    code=code,
                    symbol=MAIN_CONTRACT_MAPPING[code],
                    id=CHAOS_VIRGTOR_MAPPING[MAIN_CONTRACT_MAPPING[code]])
