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


# 已套牢投资者的平均亏损率。衡量亏损方的“痛苦指数”，亏损越深，解套压力越大。
def cp009(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    loss_chips = chip_data[chip_data['price'] > avg_price]
    if loss_chips.empty:
        alpha1 =  0.0
    else:
        avg_loss_price = np.average(loss_chips['price'],
                                weights=loss_chips['chips'])
        alpha1 = (avg_price - avg_loss_price) / (avg_loss_price + 1e-9)
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
