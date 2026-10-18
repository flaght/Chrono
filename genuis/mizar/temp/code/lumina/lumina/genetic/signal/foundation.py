import numpy as np
import six, pdb
from ultron.utilities.singleton import Singleton
from lumina.genetic.signal.method import __muster__ as signal_muster


@six.add_metaclass(Singleton)
class Signals(object):

    def __init__(self, signal_sets=None):
        self._signal_sets = signal_sets if signal_sets is not None else signal_muster
        self._init_signal()

    def _init_signal(self):
        self._function_sets = []
        for muster in self._signal_sets:
            self._function_sets.extend(muster())

    def signals_methods(self):
        return self._function_sets


signals_methods = Signals().signals_methods()
