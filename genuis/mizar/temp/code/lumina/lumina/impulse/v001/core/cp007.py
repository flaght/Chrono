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


## N期平均价格相对于全市场加权平均持仓成本（ASR）的偏离度，反映市场整体盈亏状态。
def cp007(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    asr = np.average(chip_data['price'], weights=chip_data['chips'])
    alpha1 = (avg_price - asr) / (asr + 1e-9)
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
