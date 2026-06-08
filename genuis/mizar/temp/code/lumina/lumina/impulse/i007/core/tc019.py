import pdb
from lumina.impulse.fixed import *

#tema
def tc019(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    em1 = roller_mean(close, weriod, 1, method)
    em2 = roller_mean(em1, weriod, 1, method)
    em3 = roller_mean(em2, weriod, 1, method)
    tema = 3 * em1 - 3 * em2 + em3

    alpha = roller_mean(tema, window, 1, method)
    return alpha
