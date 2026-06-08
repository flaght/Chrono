from lumina.impulse.fixed import *


#EFI
def oi034(close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    close_diff = close.diff(1)
    efi = roller_mean(close_diff * openint, weriod, 1, method)

    alpha = roller_mean(efi, window, 1, method)
    return alpha
