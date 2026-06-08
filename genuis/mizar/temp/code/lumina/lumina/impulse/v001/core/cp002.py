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
## 最高筹码峰所占的筹码百分比，值越高，该价位的支撑/阻力效应越强。
def cp002(chip_data, close):
    peak_idx = chip_data['chips'].idxmax()
    chip_peak_strength = chip_data.loc[peak_idx, 'chips']
    alpha1 =  chip_peak_strength
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha