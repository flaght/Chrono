import pdb
from lumina.impulse.fixed import *


# kst
def tc011(close, window, weriod, scalar=None, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    scalar = float(scalar) if scalar and scalar > 0 else 1000

    roc1 = (scalar * close.diff(weriod) / close.shift(weriod)) + (
        scalar * close.diff(weriod) / close.shift(weriod))
    rocma1 = roller_mean(roc1, weriod, 1, method)

    roc2 = (scalar * roc1.diff(weriod) / roc1.shift(weriod)) + (
        scalar * roc1.diff(weriod) / roc1.shift(weriod))
    rocma2 = roller_mean(roc2, weriod, 1, method)

    roc3 = (scalar * roc2.diff(weriod) / roc2.shift(weriod)) + (
        scalar * roc2.diff(weriod) / roc2.shift(weriod))
    rocma3 = roller_mean(roc3, weriod, 1, method)

    roc4 = (scalar * roc3.diff(weriod) / roc3.shift(weriod)) + (
        scalar * roc3.diff(weriod) / roc3.shift(weriod))
    rocma4 = roller_mean(roc4, weriod, 1, method)

    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    alpha = roller_mean(kst, weriod, 1, method)
    alpha = roller_mean(alpha, window, 1, method)
    return alpha
