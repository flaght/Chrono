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


# 状态因子：+1表示N期价格在峰值之上（强势区），-1表示N期价格在其之下（弱势区），0在峰值附近（博弈区）。
def cp014(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    peak_idx = chip_data['chips'].idxmax()
    chip_peak_price = chip_data.loc[peak_idx, 'price']
    alpha1 = np.sign(avg_price - chip_peak_price)
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)

    return alpha
