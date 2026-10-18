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


# 从N期平均价格到下一个显著阻力峰的价格空间百分比。值越大，说明股价上方的“真空地带”越大
def cp019(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    loss_chips = chip_data[chip_data['price'] > avg_price]
    if loss_chips.empty:
        alpha = 0.0
    else:
        resistance_idx = loss_chips['chips'].idxmax()
        next_resistance_price = loss_chips.loc[resistance_idx, 'price']

        alpha = (next_resistance_price - avg_price) / (avg_price + 1e-9)

    alpha = pd.DataFrame(np.array([alpha]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
