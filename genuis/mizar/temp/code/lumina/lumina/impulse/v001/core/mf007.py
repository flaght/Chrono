import pdb
import pandas as pd
from lumina.impulse.fixed import *


def mf007(net_in_opn, net_in_cls,  turnoverValue, window=0):
    alpha = (net_in_cls - net_in_opn) / turnoverValue
    alpha = alpha.shift(window)
    return alpha



    
