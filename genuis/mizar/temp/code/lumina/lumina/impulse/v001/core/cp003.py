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

## 筹码最密集的10%价格区间的总筹码占比。值越高，说明筹码锁定性越好，主力控盘度可能越高。
def cp003(chip_data, close, percent=0.1):
    sorted_chips = chip_data.sort_values('chips', ascending=False)
    top_percent_count = int(np.ceil(percent * len(sorted_chips)))
    top_chips = sorted_chips.head(top_percent_count)
    alpha1 =  top_chips['chips'].sum()
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
