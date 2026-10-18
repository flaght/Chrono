import numpy as np
import six, pdb
from ultron.utilities.singleton import Singleton
from lumina.genetic.strategy.method import __muster__ as strategy_muster


@six.add_metaclass(Singleton)
class Strategies(object):

    def __init__(self, strategy_sets=None):
        self._strategy_sets = strategy_sets if strategy_sets is not None else strategy_muster
        self._init_strategy()

    def _init_strategy(self):
        self._function_sets = []
        for muster in self._strategy_sets:
            self._function_sets.extend(muster())

    def strategies_methods(self):
        return self._function_sets



strategies_methods = Strategies().strategies_methods()
