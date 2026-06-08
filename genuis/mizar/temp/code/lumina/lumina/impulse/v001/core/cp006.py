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


## 收盘价下方所有筹码的总占比。衡量市场浮动盈利情况的核心指标，比例过高可能有回调压力。
def cp006(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    profit_chips = chip_data[chip_data['price'] < avg_price]
    alpha1 = profit_chips['chips'].sum()
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
