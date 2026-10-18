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


## 筹码最密集的成本价，代表市场核心成本区，通常是强支撑或阻力位。
def cp001(chip_data, close):
    peak_idx = chip_data['chips'].idxmax()
    chip_peak_price = chip_data.loc[peak_idx, 'price']
    alpha1 = chip_peak_price
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
