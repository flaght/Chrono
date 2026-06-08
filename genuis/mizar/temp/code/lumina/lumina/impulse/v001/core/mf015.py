import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf015(inflow, outflow, netFlow, window=0):
    total_activity = inflow - outflow

    alpha = 1 - abs(netFlow) / (total_activity + 1e-6)
    alpha = alpha.shift(window)
    return alpha


