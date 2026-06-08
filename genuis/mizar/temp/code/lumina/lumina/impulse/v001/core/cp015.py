import pandas as pd
import numpy as np
from lumina.impulse.fixed import *
'''
chips_data:
        price     chips
7     8.12535  1.642764
12    8.25060  0.113686
13    8.27565  0.174279
15    8.32575  1.801554
'''


# N期收盘价所在区间的筹码密度。值越高，说明当前价位是多空争夺焦点，正在激烈换手
def cp015(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    closest_price_idx = (chip_data['price'] - avg_price).abs().idxmin()
    chip_peak_price = chip_data.loc[closest_price_idx, 'price']

    alpha = pd.DataFrame(np.array([chip_peak_price]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
