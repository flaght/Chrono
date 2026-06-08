import pdb
from lumina.impulse.fixed import *

# PPO
def tc014(close, window, fast, slow, scalar=None, ewm=False):
    if slow < fast:
        fast, slow = slow, fast
    method = 'ewm' if ewm else 'rolling'
    scalar = float(scalar) if scalar else 10000
    fast_ema = roller_mean(close, fast, 1, method)
    slow_ema = roller_mean(close, slow, 1, method)
    ppo = scalar * (fast_ema - slow_ema) / slow_ema

    alpha = roller_mean(ppo, window, 1, method)
    return alpha
