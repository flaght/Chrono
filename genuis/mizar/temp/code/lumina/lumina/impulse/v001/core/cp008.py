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


# 已获利投资者的平均利润率。衡量盈利方的“安全垫”厚度，利润越高，锁定意愿可能越强
def cp008(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    profit_chips = chip_data[chip_data['price'] < avg_price]
    if profit_chips.empty:
        return 0.0
    avg_profit_price = np.average(profit_chips['price'],
                                  weights=profit_chips['chips'])
    alpha1 = (avg_price - avg_profit_price) / (avg_profit_price + 1e-9)
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
