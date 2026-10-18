import pdb
from lumina.impulse.fixed import *

#BIAS
def tc006(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    bma = roller_mean(close, weriod, 1, method)
    alpha = (close - bma) / bma
    alpha = roller_mean(alpha, window, 1, method)
    return alpha