import pdb
from lumina.impulse.fixed import *


def tf008(open, high, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    core1 = roller_corr(volume, high, weriod, weriod, method)
    core1 = np.minimum(np.log(np.arctan(volume).mul(core1)), open.diff(6))

    alpha = roller_mean(core1, window, window, method)

    return alpha
