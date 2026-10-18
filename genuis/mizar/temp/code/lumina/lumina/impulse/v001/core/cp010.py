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


# 获利盘总量与套牢盘总量的比值。远大于1，多头主导；远小于1，空头（套牢盘）主导。
def cp010(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    profit_chips = chip_data[chip_data['price'] < avg_price]
    profit_ratio = profit_chips['chips'].sum()
    alpha1 = profit_ratio / (100 - profit_ratio + 1e-9)
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
