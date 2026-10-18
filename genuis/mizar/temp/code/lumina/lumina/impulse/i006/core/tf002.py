import pdb
from lumina.impulse.fixed import *


def tf002(open, high, close, volume, value, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    corr1 = roller_corr(open.diff(2), volume.diff(2), weriod, weriod, method)

    corr2 = roller_corr(high.diff(2), value.diff(2), weriod, weriod, method)

    corr3 = roller_corr((1 / (1 + np.exp(-close.diff(2)))), corr1, weriod,
                        weriod, method)

    core1 = corr3.sub(corr2)

    alpha = roller_mean(core1, window, window, method)

    return alpha
