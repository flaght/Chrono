import pandas as pd
from lumina.impulse.fixed import *
'''
chips_data:
        price     chips
7     8.12535  1.642764
12    8.25060  0.113686
13    8.27565  0.174279
15    8.32575  1.801554
'''


## 当前价格与最强筹码峰的偏离程度。正值越大，代表股价已有效脱离主要成本区。
def cp011(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    peak_idx = chip_data['chips'].idxmax()
    chip_peak_price = chip_data.loc[peak_idx, 'price']
    alpha1 = (avg_price - chip_peak_price) / (chip_peak_price + 1e-9)
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
