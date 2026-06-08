import pandas as pd
import numpy as np
from lumina.impulse.fixed import *
from scipy.signal import find_peaks
'''
chips_data:
        price     chips
7     8.12535  1.642764
12    8.25060  0.113686
13    8.27565  0.174279
15    8.32575  1.801554
'''


# N期价格上方第一个显著筹码峰的强度。量化了股价上涨时，即将遇到的第一个“拦路虎”有多强。
def cp017(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    loss_chips = chip_data[chip_data['price'] > avg_price]
    if loss_chips.empty:
        alpha1 = 0.0
    else:
        peaks, _ = find_peaks(loss_chips['chips'])
        if len(peaks) == 0:
            alpha1 = loss_chips['chips'].max(
            )  # Fallback to max if no clear peak
        else:
            alpha1 = loss_chips['chips'].iloc[peaks].max()

    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
