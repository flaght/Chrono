import pandas as pd
from lumina.impulse.fixed import *
from scipy.signal import find_peaks
'''
chips_data:
        price     chips
7     8.12535  1.642764
12    8.25060  0.113686
13    8.27565  0.174279
15    8.32575  1.801554
'''


# 最近支撑与最近阻力的强度比。大于1说明下方支撑强于上方阻力，股价易涨难跌。
def cp018(chip_data, close, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    avg_price = roller_mean(close, weriod, 1, method).values[-1][0]
    profit_chips = chip_data[chip_data['price'] < avg_price]
    loss_chips = chip_data[chip_data['price'] > avg_price]
    support = 0.0
    resistance = 0.0
    if not profit_chips.empty:
        peaks, _ = find_peaks(profit_chips['chips'])
        if len(peaks) == 0:
            support = profit_chips['chips'].max(
            )  # Fallback to max if no clear peak
        else:
            support = profit_chips['chips'].iloc[peaks].max()

    if not loss_chips.empty:
        peaks, _ = find_peaks(loss_chips['chips'])
        if len(peaks) == 0:
            resistance = loss_chips['chips'].max(
            )  # Fallback to max if no clear peak
        else:
            resistance = loss_chips['chips'].iloc[peaks].max()
    alpha = support / (resistance + 1e-9)
    alpha = pd.DataFrame(np.array([alpha]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
