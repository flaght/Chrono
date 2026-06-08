import pdb
from lumina.impulse.fixed import *

# T3
def tc018(close, window, weriod, ewm=False):
    a = 0.7
    c1 = -a * a**2
    c2 = 3 * a**2 + 3 * a**3
    c3 = -6 * a**2 - 3 * a - 3 * a**3
    c4 = a**3 + 3 * a**2 + 3 * a + 1
    method = 'ewm' if ewm else 'rolling'
    e1 = roller_mean(close, weriod, 1, method)
    e2 = roller_mean(e1, weriod, 1, method)
    e3 = roller_mean(e2, weriod, 1, method)
    e4 = roller_mean(e3, weriod, 1, method)
    e5 = roller_mean(e4, weriod, 1, method)
    e6 = roller_mean(e5, weriod, 1, method)

    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    alpha = roller_mean(t3, window, 1, method)
    return alpha
