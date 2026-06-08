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


# N期平均价格下方第一个显著筹码峰的强度。量化了股价回调时，最近的“护城河”有多深。
def cp016(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    profit_chips = chip_data[chip_data['price'] < avg_price]
    if profit_chips.empty:
        alpha1 = 0.0
    peaks, properties = find_peaks(profit_chips['chips'])
    if len(peaks) == 0:
        alpha1 = profit_chips['chips'].max(
        )  # Fallback to max if no clear peak
    else:
        alpha1 = np.max(properties['peak_heights'])
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
